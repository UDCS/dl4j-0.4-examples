package org.deeplearning4j.examples.convolution;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CharacterRecognizer extends JFrame{

	static final List<String> LABELS = Arrays.asList("circle", "fillCircle", "fillSquare", "line", "square");
	private static final String COEFF_FILE = "/home/bmccutchon/git/dl4j-0.4-examples/coefficients.bin";
	private static final String CONF_FILE = "/home/bmccutchon/git/dl4j-0.4-examples/conf.json";
	private static final long serialVersionUID = 7725679730437636788L;
	private static final Logger log = LoggerFactory.getLogger(CharacterRecognizer.class);
	static BufferedImage img;	
	private static final int TRAIN = 1;
	private static final int TEST = 2;
	private static final int BOTH = TRAIN | TEST;


	public static void main(String[] args) throws IOException, InterruptedException {
		int action = BOTH;
		//generateShapeImages(action);
		//generateCharacterImages(action);
		MultiLayerNetwork model = null;
		if((action & TRAIN) != 0)  model = trainNetwork("/home/share/vision2014/NeuralNets/ShapesTrain/");
		if((action & TEST)  != 0)  predict(model, "/home/share/vision2014/NeuralNets/ShapesTest/");
	}



	private static MultiLayerNetwork trainNetwork(String path) throws IOException, InterruptedException{
		int nChannels = 1;
		int outputNum = 5;
		int batchSize = 16;
		int nEpochs = 12;
		int iterations = 1;
		int seed = 123;

		System.out.println("Training");

		/*
		//traverse dataset to get each label
		List<String> labels = new ArrayList<>(); 
		for(File f : new File(path).listFiles()) { 
			labels.add(f.getName());
		}  
		 */
		List<String> labels = new ArrayList<>(LABELS);

		// Instantiating RecordReader. Specify height and width of images.
		RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

		// Point to data path. 
		recordReader.initialize(new FileSplit(new File(path), new Random(123)));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());

		// Configure the network
		log.info("Build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0005)
				.learningRate(0.01)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.list(4)
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						.nIn(nChannels)
						.stride(1, 1)
						.nOut(20).dropOut(0.5)
						.activation("relu")
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(2, new DenseLayer.Builder().activation("relu")
						.nOut(500).build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation("softmax")
						.build())
				.backprop(true).pretrain(false);
		new ConvolutionLayerSetup(builder,28,28,1);

		// Build the network
		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();



		//model.fit(iter);


		//iter = new MnistDataSetIterator(batchSize,true,12345);

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for( int i=0; i < nEpochs; i++ ) {
			while(iter.hasNext()){
				DataSet next = iter.next(batchSize);
				next.normalize();
				model.fit(next);
			}

			log.info("*** Completed epoch {} ***", i);
			iter.reset();
		}
		log.info("****************Example finished********************");



		//Write the network parameters:
		System.out.println("Saving model");
		try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(
				Paths.get(COEFF_FILE)))){
			Nd4j.write(model.params(), dos);
		}

		//Write the network configuration:
		FileUtils.write(new File(CONF_FILE), model.getLayerWiseConfigurations().toJson());

		return model;
	}


	private static void predict( MultiLayerNetwork model, String path) throws IOException, InterruptedException{
		//create array of strings called labels, read from the subdirectories of the directory below
		System.out.println("Predicting");


		/*
		//traverse dataset to get each label
		List<String> labels = new ArrayList<>(); 
		for(File f : new File(path).listFiles()) { 
			labels.add(f.getName());
		}  
		 */
		List<String> labels = LABELS;

		// Instantiating RecordReader. Specify height and width of images.
		RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

		// Point to data path. 
		recordReader.initialize(new FileSplit(new File(path)));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());



		//Load network configuration from disk, if needed
		if(model == null){
			MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.
					readFileToString(new File(CONF_FILE)));

			//Load parameters from disk:
			INDArray newParams;
			try(DataInputStream dis = new DataInputStream(new FileInputStream(COEFF_FILE))){
				newParams = Nd4j.read(dis);
			}

			//Create a MultiLayerNetwork from the saved configuration and parameters
			model = new MultiLayerNetwork(confFromJson);
			model.init();
			model.setParameters(newParams);
		}

		//iter = new MnistDataSetIterator(64, false, 12345);


		Evaluation eval = new Evaluation();
		while(iter.hasNext()){
			DataSet next = iter.next();
			next.normalize();
			INDArray predict = model.output(next.getFeatureMatrix());
			eval.eval(next.getLabels(), predict);
		}

		System.out.println(eval.stats());
	}



	private static void generateCharacterImages(int trainOrTest){
		img = new BufferedImage(HEIGHT, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
		JPanel panel;
		final int HEIGHT = 28;
		final int WIDTH = 28;

		CharacterRecognizer cd = new CharacterRecognizer();
		panel = new JPanel();
		//cd.setContentPane(panel);
		cd.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		cd.setPreferredSize(new Dimension(400, 400));
		cd.setLocation(2000, 0);
		cd.setVisible(true);
		cd.pack();


		char[] letters = {'a', 'e', 'i', 'o', 'u'};
		String[] names = {Font.DIALOG, Font.DIALOG_INPUT/*, Font.MONOSPACED, Font.SERIF, Font.SANS_SERIF*/};
		int[]   styles = {/*Font.BOLD, Font.ITALIC, */Font.PLAIN};


		img = new BufferedImage(HEIGHT, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = (Graphics2D)img.createGraphics();
		g.setBackground(Color.BLACK);
		AffineTransform AT = g.getTransform();

		double rotOffset = 0.0;
		double skewOffset = 0.0;
		for(int i : new int[] {TEST}){
			if((i & trainOrTest) == 0)
				continue;
			if(i == TEST){	// Different set for testing than training
				rotOffset = 0.005;
				skewOffset = 0.002;
			}
			int scoringLabel = -1;	// Each character gets a consecutive int label
			int fileNum = 0;
			for(char c : letters){
				scoringLabel++;
				for(String n : names){
					for(int s : styles){
						for(double rot = -.05 + rotOffset; rot <= .050; rot += .01){
							for(double skew = -0.01 + skewOffset; skew <= 0.01; skew += 0.005){
								g.setTransform(AT);
								System.out.printf("Font: %c %s %d %f %f\n", c, n, s, rot, skew);
								Font f = new Font(n, s, HEIGHT);
								g.setFont(f);
								g.transform(new AffineTransform(1, 0, 0, 1, -WIDTH/2, -HEIGHT/2));
								g.transform(new AffineTransform(Math.cos(rot), -Math.sin(rot) + skew/100,
										Math.sin(rot), Math.cos(rot) + skew/100, 0, 0));
								g.transform(new AffineTransform(1, 0, 0, 1, WIDTH/2, HEIGHT/2));
								g.setColor(Color.BLACK);
								g.fillRect(-20, -20, 200, 200);
								g.setColor(Color.WHITE);
								g.drawString("" + c, 4, HEIGHT-4);
								//cd.repaint();
								try {
									Thread.sleep(000);
								} catch (InterruptedException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}

								File outputfile;
								if(i == TRAIN)
									outputfile = new File("/home/share/vision2014/NeuralNets/CharsTrain/" + c + 
											"/" + fileNum + ".jpg");
								else
									outputfile = new File("/home/share/vision2014/NeuralNets/CharsTest/" + c + 
											"/" + fileNum + ".jpg");

								fileNum++;
								//try {
								//	if(Math.random() > -0.7)
								//		ImageIO.write(img, "jpg", outputfile);
								//} catch (IOException e) {
								//	e.printStackTrace();
								//}

							}
						}
					}
				}
			}
		}
	}



	private static void generateShapeImages(int trainOrTest){
		img = new BufferedImage(HEIGHT, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
		JPanel panel;
		final int HEIGHT = 28;
		final int WIDTH = 28;

		CharacterRecognizer cd = new CharacterRecognizer();
		panel = new JPanel();
		//cd.setContentPane(panel);
		cd.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		cd.setPreferredSize(new Dimension(400, 400));
		cd.setLocation(2000, 0);
		cd.setVisible(true);
		cd.pack();

		img = new BufferedImage(HEIGHT, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = (Graphics2D)img.createGraphics();
		g.setBackground(Color.BLACK);
		AffineTransform AT = g.getTransform();


		double rotOffset = 0.0;
		double skewOffset = 0.0;
		for(int i : new int[] {TRAIN, TEST}){
			if((i & trainOrTest) == 0)
				continue;
			if(i == TEST){	// Different set for testing than training
				rotOffset = 0.005;
				skewOffset = 0.001;
			}
			int scoringLabel = -1;	// Each character gets a consecutive int label
			int fileNum = 0;
			String[] sh = {"circle", "fillCircle", "square", "fillSquare", "line"};
			for(int shape = 0; shape < 5; shape++){
				for(double rot = -.09 + rotOffset; rot <= .09; rot += .01){
					for(double skew = -0.01 + skewOffset; skew <= 0.01; skew += 0.002){
						g.setTransform(AT);
						g.transform(new AffineTransform(1, 0, 0, 1, -WIDTH/2, -HEIGHT/2));
						g.transform(new AffineTransform(Math.cos(rot), -Math.sin(rot) + skew/100,
								Math.sin(rot), Math.cos(rot) + skew/100, 0, 0));
						g.transform(new AffineTransform(1, 0, 0, 1, WIDTH/2, HEIGHT/2));
						g.setColor(Color.BLACK);
						g.fillRect(-20, -20, 200, 200);
						g.setColor(Color.WHITE);
						switch(shape){
						case 0:
							g.drawOval(4, 4, 20, 20);
							break;
						case 1:
							g.fillOval(4, 4, 20, 20);
							break;
						case 2:
							g.drawRect(4, 4, 20, 20);
							break;
						case 3:
							g.fillRect(4, 4, 20, 20);
							break;
						case 4:
							g.drawLine(4, 4, 20, 20);
							break;
						}
						cd.repaint();
						try {
							Thread.sleep(005);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}

						File outputfile;
						if(i == TRAIN)
							outputfile = new File("/home/share/vision2014/NeuralNets/ShapesTrain/" + sh[shape] + 
									"/" + fileNum + ".jpg");
						else
							outputfile = new File("/home/share/vision2014/NeuralNets/ShapesTest/" + sh[shape] + 
									"/" + fileNum + ".jpg");
						fileNum++;
						try {
							if(Math.random() > 0.7)
								ImageIO.write(img, "jpg", outputfile);
						} catch (IOException e) {
							e.printStackTrace();
						}

					}
				}
			}
		}
	}

	@Override
	public void paint(Graphics g) {
		super.paint(g);
		g.drawImage(img, 20, 20, img.getWidth(), img.getHeight(), null);
	}
}
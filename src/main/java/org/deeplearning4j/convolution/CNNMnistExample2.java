package org.deeplearning4j.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by willow on 5/11/15.
 */
public class CNNMnistExample2 {

    private static final Logger log = LoggerFactory.getLogger(CNNMnistExample2.class);
    private static final int numRows = 28;
    private static final int numColumns = 28;

    @Option(name="-samples", usage="number of samples to get")
    private static int numSamples = 100;

    @Option(name="-batch", usage="batch size for training" )
    private static int batchSize = 10;

    @Option(name="-featureMap", usage="size of feature map. Just enter single value")
    int featureMapSize = 5;

    @Option(name="-learningRate", usage="learning rate")
    private static double learningRate = 0.10;

    @Option(name="-activate", usage="activation function")
    private static String activationFunc="sigmoid";

    @Option(name="-loss", usage="loss function")
    private static LossFunctions.LossFunction lossFunc = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

    @Option(name="-hLayerSize", usage="hidden layer size")
    private static int hLayerSize = 18;

    private static double numTrainSamples = numSamples * 0.8;
    private static double numTestSamples = numSamples - numTrainSamples;

    static DataSetIterator loadData(int numTrainSamples) throws Exception{
        //SamplingDataSetIterator - TODO make sure representation of each classification in each batch
        DataSetIterator dataIter = new MnistDataSetIterator(batchSize, numTrainSamples);
        return dataIter;
    }

    static MultiLayerNetwork buildModel(final int featureMapSize){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns)
                .nOut(10)
                .batchSize(batchSize)
                .iterations(100)
                .weightInit(WeightInit.ZERO)
                .activationFunction(activationFunc)
                .filterSize(batchSize, 1, numRows, numColumns)
                .lossFunction(lossFunc).learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .hiddenLayerSizes(hLayerSize)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .preProcessor(1, new ConvolutionPostProcessor())
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(featureMapSize, featureMapSize);
                    }
                })
                .override(1, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new SubsamplingLayer());
                    }
                })
                .override(2, new ClassifierOverride(2))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));
        return model;
    }

    static MultiLayerNetwork trainModel(DataSetIterator data, MultiLayerNetwork model){
        while (data.hasNext()){
            DataSet allData = data.next();
            allData.normalizeZeroMeanZeroUnitVariance();
            model.fit(allData);
        }
        return model;

    }

    static void evaluateModel(DataSetIterator testData, MultiLayerNetwork model){
        Evaluation eval = new Evaluation();
        DataSet allTest = testData.next();

        INDArray testInput = allTest.getFeatureMatrix();
        INDArray testLabels = allTest.getLabels();
        INDArray output = model.output(testInput);
        eval.eval(testLabels, output);
        log.info(eval.stats());
    }

    public void exec(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);

        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        log.info("Load data....");
        DataSetIterator dataIter = loadData((int) numTrainSamples);

        log.info("Build model....");
        MultiLayerNetwork model = buildModel(featureMapSize);

        log.info("Train model....");
        trainModel(dataIter, model);

        log.info("Evaluate model....");
        DataSetIterator testData = loadData((int) numTestSamples);
        evaluateModel(testData, model);

        log.info("****************Example finished********************");
    }

    public static void main( String[] args ) throws Exception
    {
        new CNNMnistExample2().exec(args);

    }
}
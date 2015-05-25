package org.deeplearning4j.convolution;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Created by willow on 5/11/15.
 */
public class CNNMnistExample {

    private static final Logger log = LoggerFactory.getLogger(CNNMnistExample.class);

    public static void main(String[] args) throws Exception {

        final int numRows = 28;
        final int numColumns = 28;
        int batchSize = 100;
        int numSamples = 1000;
        double numTrainSamples = numSamples * 0.8;

        log.info("Load data....");
        DataSetIterator mnist = new MnistDataSetIterator(numSamples,numSamples); // TODO there are 60k avail
        DataSet allMnist = mnist.next();
        allMnist.normalizeZeroMeanZeroUnitVariance();
//        allMnist.normalize();

        log.info("Split data....");
        SplitTestAndTrain trainTest = allMnist.splitTestAndTrain(((int) numTrainSamples)); // train set that is the result - should flip // TODO put back to 80% of data
        DataSet trainInput = trainTest.getTrain(); // get feature matrix and labels for training
        INDArray testInput = trainTest.getTest().getFeatureMatrix();
        INDArray testLabels = trainTest.getTest().getLabels();

        DataSetIterator trainDataSetIterator = new SamplingDataSetIterator(trainInput, batchSize, (int) numTrainSamples);

        log.info("Build model....");
        // TODO iterations back to 10
        // TODO try the if then without the input and preprocess for all layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns).nOut(10).batchSize(batchSize)
                .iterations(100).weightInit(WeightInit.ZERO)
                .activationFunction("sigmoid").filterSize(30, 1, numRows, numColumns)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).learningRate(0.13)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT).constrainGradientToUnitNorm(true)
                .list(3).hiddenLayerSizes(72)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns)).preProcessor(1, new ConvolutionPostProcessor())
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(11, 11);
                    }
                }).override(1, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new SubsamplingLayer());
                    }
                }).override(2, new ClassifierOverride(2))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        log.info("Train model....");
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1)));

        while(trainDataSetIterator.hasNext()) {
            DataSet trainBatchData = trainDataSetIterator.next();
            network.fit(trainBatchData);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = network.output(testInput);
        eval.eval(testLabels, output);
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }
}
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
        int batchSize = 10; // TODO return to 1802

        log.info("Load data....");
        DataSetIterator mnist = new MnistDataSetIterator(100,100); // TODO change back to 100,1000 (batch, numExamples) - there are 60k avail
        DataSet all = mnist.next();
        all.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data....");
        SplitTestAndTrain trainTest = all.splitTestAndTrain(90); // train set that is the result - should flip // TODO put back to 80% of data
        DataSet trainInput = trainTest.getTrain(); // get feature matrix and labels for training
        INDArray testInput = trainTest.getTest().getFeatureMatrix();
        INDArray testLabels = trainTest.getTest().getLabels();

        log.info("Build model....");
        // TODO iterations back to 10
        // TODO try the if then without the input and preprocess for all layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns).nOut(10).batchSize(batchSize)
                .iterations(4).weightInit(WeightInit.VI)
                .activationFunction("sigmoid").filterSize(7, 1, numRows, numColumns)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS).constrainGradientToUnitNorm(true)
                .list(3).hiddenLayerSizes(new int[]{50})
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns)).preProcessor(1, new ConvolutionPostProcessor())
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(9, 9);
                    }
                }).override(1, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new SubsamplingLayer());
                    }
                }).override(2, new ClassifierOverride())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        log.info("Train model....");
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1)));
        network.fit(trainInput);

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = network.output(testInput);
        eval.eval(testLabels,output);
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }
}
package org.deeplearning4j.convolution;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.ConvolutionDownSampleLayer;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author sonali
 */
public class CNNIrisExample {

    private static Logger log = LoggerFactory.getLogger(CNNIrisExample.class);

    public static void main(String[] args) {

        int batchSize = 110;
        final int numRows = 2;
        final int numColumns = 2;
        /**
         *Set a neural network configuration with multiple layers
         */
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .iterations(10).weightInit(WeightInit.VI)
                .activationFunction("tanh").filterSize(5, 1, numRows, numColumns).constrainGradientToUnitNorm(true)
                .nIn(numRows * numColumns).nOut(3).batchSize(batchSize)
                .dropOut(0.5)
                .list(2)
                .preProcessor(1, new ConvolutionPostProcessor()).inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .hiddenLayerSizes(9)
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(2, 2);
                    }
                }).override(1, new ClassifierOverride())
                .build();

        //Create a neural net from the configuration
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        //Iterator for iris dataset
        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        //Normalize data
        org.nd4j.linalg.dataset.DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        //Split data into testing and training
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);

        //Train model
        network.fit(trainTest.getTrain());

        //Evaluate results of model
        Evaluation eval = new Evaluation();
        INDArray output = network.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(),output);
        log.info("Score " +eval.stats());
    }
}

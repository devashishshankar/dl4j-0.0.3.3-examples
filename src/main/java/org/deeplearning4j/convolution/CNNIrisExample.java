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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;

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
        log.info("Load data....");
        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        DataSet iris = irisIter.next();
        iris.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain trainTest = iris.splitTestAndTrain(batchSize);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns)
                .nOut(3)
                .iterations(10)
                .weightInit(WeightInit.VI)
                .activationFunction("tanh")
                .filterSize(5, 1, numRows, numColumns)
                .batchSize(batchSize)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .dropOut(0.5)
                .list(2)
                .hiddenLayerSizes(4)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .preProcessor(0, new ConvolutionPostProcessor())
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(2, 2);
                    }
                })
                .override(1, new ClassifierOverride(1))
                .build();

        log.info("Build model....");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(trainTest.getTrain());

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(), output);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}

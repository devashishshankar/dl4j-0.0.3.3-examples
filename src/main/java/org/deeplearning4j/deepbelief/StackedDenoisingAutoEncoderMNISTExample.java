package org.deeplearning4j.deepbelief;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * @author devashish.shankar
 * @version 1.0, 6/5/15
 */
public class StackedDenoisingAutoEncoderMNISTExample {
    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoderMNISTExample.class);

    public static void main(String[] args) throws Exception {
        log.info("Load data....");
        DataSetIterator iter = new MultipleEpochsIterator(15, new MnistDataSetIterator(100,60000));
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new AutoEncoder()).corruptionLevel(0.1)
                .nIn(784)
                .nOut(10)
                .weightInit(WeightInit.VI)
                .constrainGradientToUnitNorm(true)
                .iterations(1)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(0.001)
                .list(4)
                .hiddenLayerSizes(new int[]{1000, 1000, 1000})
                .override(3,new ClassifierOverride())
                .build();

        log.info("Build model....");

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();

        DataSetIterator testIter = new MnistDataSetIterator(100,10000);
        while(testIter.hasNext()) {
            DataSet testMnist = testIter.next();
            testMnist.normalizeZeroMeanZeroUnitVariance();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}

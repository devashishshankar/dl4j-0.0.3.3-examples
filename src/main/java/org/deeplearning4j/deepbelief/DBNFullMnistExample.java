package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNFullMnistExample {

    private static Logger log = LoggerFactory.getLogger(DBNFullMnistExample.class);


    public static void main(String[] args) throws Exception {

        log.info("Load data....");
        Nd4j.dtype = DataBuffer.Type.FLOAT;
        DataSetIterator iter = new MnistDataSetIterator(100,60000);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM()).nIn(784).nOut(10).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 1)).iterations(5)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT).learningRate(1e-1f)
                .momentum(0.5).momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4).hiddenLayerSizes(new int[]{500, 250, 200})
                .override(3,new ClassifierOverride(3))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

        log.info("Train model....");
        while(iter.hasNext()) {
            DataSet next = iter.next();
            model.fit(next);
        }
        iter.reset();

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        while(iter.hasNext()) {
            DataSet d2 = iter.next();
            INDArray predict2 = model.output(d2.getFeatureMatrix());
            eval.eval(d2.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}

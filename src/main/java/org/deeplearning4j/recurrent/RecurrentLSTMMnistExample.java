package org.deeplearning4j.recurrent;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
//import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.Arrays;
import java.util.Collections;

/**
 * Created by willow on 5/11/15.
 */

public class RecurrentLSTMMnistExample {

    private static Logger log = LoggerFactory.getLogger(RecurrentLSTMMnistExample.class);

    public static void main(String[] args) throws Exception {

        log.info("Loading data...");
        MnistDataFetcher fetcher = new MnistDataFetcher(true);

        log.info("Building model...");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("sigmoid")
                .layer(new LSTM())
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .constrainGradientToUnitNorm(true)
                .nIn(784).nOut(784).build();
        Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);

        log.info("Training model...");
        for(int i=0 ; i < 100; i++) {
            fetcher.fetch(100);
            DataSet mnist = fetcher.next();
            layer.fit(mnist.getFeatureMatrix());
        }

    // Generative Model - unsupervised and its time series based which requires different evaluation technique

    }

}

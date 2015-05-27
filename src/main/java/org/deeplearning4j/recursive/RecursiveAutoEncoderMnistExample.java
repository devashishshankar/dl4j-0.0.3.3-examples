package org.deeplearning4j.recursive;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Created by willow on 5/11/15.
 */
public class RecursiveAutoEncoderMnistExample {

    private static Logger log = LoggerFactory.getLogger(RecursiveAutoEncoderMnistExample.class);

    public static void main(String[] args) throws Exception {

        log.info("Loading data...");
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(10);
        DataSet d2 = fetcher.next();
        INDArray input = d2.getFeatureMatrix();

        log.info("Building model...");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .corruptionLevel(0.3).weightInit(WeightInit.VI)
                .iterations(10)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(784).nOut(600)
                .layer(new RecursiveAutoEncoder())
                .build();

        log.info("Training model...");
        Layer model = LayerFactories.getFactory(conf).create(conf);
        model.setIterationListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1)));
        model.setParams(model.params());
        model.fit(input);

        // Generative Model - unsupervised and requires different evaluation technique

    }
}
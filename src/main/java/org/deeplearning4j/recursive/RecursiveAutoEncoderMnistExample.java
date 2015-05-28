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
import java.util.Collections;

/**
 * Created by willow on 5/11/15.
 */
public class RecursiveAutoEncoderMnistExample {

    private static Logger log = LoggerFactory.getLogger(RecursiveAutoEncoderMnistExample.class);

    public static void main(String[] args) throws Exception {

        log.info("Loading data...");
        MnistDataFetcher fetcher = new MnistDataFetcher(true);

        log.info("Building model...");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RecursiveAutoEncoder())
                .nIn(784)
                .nOut(600)
                .momentum(0.9f)
                .corruptionLevel(0.3)
                .weightInit(WeightInit.VI)
                .constrainGradientToUnitNorm(true)
                .iterations(10)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .learningRate(1e-1f)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));
        model.setParams(model.params());

        log.info("Training model...");

        for(int i=0 ; i < 100; i++) {
            fetcher.fetch(100);
            DataSet data = fetcher.next();
            INDArray input = data.getFeatureMatrix();
            model.fit(input);
        }

        // Generative Model - unsupervised and requires different evaluation technique

    }
}
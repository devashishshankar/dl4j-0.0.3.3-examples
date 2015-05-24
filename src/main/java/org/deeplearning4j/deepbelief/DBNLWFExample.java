package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * Created by agibsonccc on 10/2/14.
 **/
public class DBNLWFExample {
    private static Logger log = LoggerFactory.getLogger(DBNLWFExample.class);


    public static void main(String[] args) throws Exception {

        log.info("Load data....");
        DataSetIterator fetcher = new LFWDataSetIterator(28,28);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM()).nIn(fetcher.inputColumns()).nOut(fetcher.totalOutcomes())
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-3f)
                .list(4).hiddenLayerSizes(new int[]{600, 250, 200})
                .override(3,new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(Arrays.asList((IterationListener) new NeuralNetPlotterIterationListener(1)));

        log.info("Train model....");
        while(fetcher.hasNext()) {
            DataSet next = fetcher.next();
            next.normalizeZeroMeanZeroUnitVariance();
            model.fit(next);

        }



    }


}

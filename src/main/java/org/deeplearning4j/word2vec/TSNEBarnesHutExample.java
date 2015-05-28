package org.deeplearning4j.word2vec;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Barnes-Hut better for large real-world datasets
 * Pass in words.txt at CLI
 */
public class TSNEBarnesHutExample {

    private static Logger log = LoggerFactory.getLogger(TSNEBarnesHutExample.class);

    public static void main(String[] args) throws Exception {

        log.info("Load & vectorize data....");
        WeightLookupTable pair = SerializationUtils.readObject(new File(args[0]));

        log.info("Build model....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(10000)
                .normalize(true)
                .stopLyingIteration(250)
                .learningRate(500)
                .theta(0.5)
                .setMomentum(0.5)
                .useAdaGrad(false)
                .usePca(false)
                .build();

        log.info("Plot Vocab TSNE....");
        pair.plotVocab(tsne);

    }

}

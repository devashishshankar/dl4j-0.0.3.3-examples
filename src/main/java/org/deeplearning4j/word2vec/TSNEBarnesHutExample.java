package org.deeplearning4j.word2vec;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Barnes-Hut better for large real-world datasets
 * Pass in words.txt at CLI
 */
public class TSNEBarnesHutExample {

    private static Logger log = LoggerFactory.getLogger(TSNEBarnesHutExample.class);

    public static void main(String[] args) throws Exception {
        List<String> cacheList = new ArrayList<>();

        log.info("Load & vectorize data....");
        Pair<InMemoryLookupTable,VocabCache> pair = WordVectorSerializer.loadTxt(new File(args[0]));
        VocabCache vocabCache = pair.getSecond();
        INDArray weights = pair.getFirst().getSyn0();

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

        for(int i = 0; i < vocabCache.numWords(); i++)
            cacheList.add(vocabCache.wordAtIndex(i));

        log.info("Plot Vocab TSNE....");
        tsne.plot(weights, 2, cacheList);

    }

}

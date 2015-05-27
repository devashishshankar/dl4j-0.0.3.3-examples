package org.deeplearning4j.word2vec;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/20/14.
 */
public class TSNECreateDataExample {

    private static Logger log = LoggerFactory.getLogger(TSNECreateDataExample.class);

    public static void main(String[] args) throws Exception  {

        List<String> cacheList = new ArrayList<>();

        log.info("Build model....");
        Tsne tsne = new Tsne.Builder()
                .setMaxIter(10000)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
                .usePca(false)
                .build();

        log.info("Vectorize data....");
        Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(new File(args[0]));
        VocabCache cache = vectors.getSecond();
        INDArray weights = vectors.getFirst().getSyn0();

        for(int i = 0; i < cache.numWords(); i++)
            cacheList.add(cache.wordAtIndex(i));

        log.info("Plot TSNE....");
        tsne.plot(weights,2,cacheList);
    }



}

package org.deeplearning4j.recursive;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.CSVDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.rntn.RNTN;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;


/**
 * Recursive Neural Tensor Network (RNTN)
 *
 * Created by willow on 5/11/15.
 *
 * Currently in development in progress - not complete yet
 */
public class RNTNTwitterExample {

    private static final Logger log = LoggerFactory.getLogger(RNTNTwitterExample.class);
    String fileName = "resources/sentiment-tweets.small.csv";
//    String fileName = "tweets_clean.txt";

    private static int batchSize = 10;
    private static int numSamples = 100;
    private static int labelColumn = 1;

    static DataSetIterator loadData(String fileName)  throws Exception {
        InputStream lwfIs = new FileInputStream(new File(fileName));
//        InputStream lwfIs = new ClassPathResource(fileName).getFile();
        DataSetIterator dataIter = new CSVDataSetIterator(batchSize, numSamples, lwfIs, labelColumn);
//        CSVRecordReader data = new CSVRecordReader().initialize(new FileSplit(new File(fileName)));
//        Collection<Writable> next = data.next();

        return dataIter;
    }

    static Word2Vec buildVectors() {

        //        Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(new File(fileName));
        // use word2vec as a lookup - feed rntn consitnuency tables - parse - sentence iterator that iterates through corpus
        // get corpus and feed into sentence iterator, fit vectors, loop and fit rntn



//        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(new File(fileName));

//        RNTN t = new RNTN.Builder()
//                .setActivationFunction(Activations.hardTanh()).setFeatureVectors(fetcher.getVec())
//                .setUseTensors(true).build();
//
//        TreeVectorizer vectorizer = new TreeVectorizer(new TreeParser());
        TreeVectorizer vectorizer = new TreeVectorizer();

        Word2Vec vec;

        vec = new Word2Vec.Builder()
                .vocabCache(cache).index(index)
                .iterate(iter).tokenizerFactory(tokenizerFactory)
                .lookupTable(lookupTable).build();
        vec.fit();


        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory(false);
        VocabCache cache = new InMemoryLookupCache();
        InvertedIndex index = new LuceneInvertedIndex.Builder()
                .indexDir(new File("rntn-index")).cache(cache).build();
        WeightLookupTable lookupTable = new InMemoryLookupTable.Builder().cache(cache)
                .vectorLength(100).build();
    }

    static RNTN buildModel(Word2Vec vec) {
        RNTN rntn = new RNTN.Builder()
                .setActivationFunction("tanh")
                .setAdagradResetFrequency(1)
                .setCombineClassification(true)
                .setFeatureVectors(vec)
                .setRandomFeatureVectors(false)
                .setUseTensors(false)
                .build();
        return rntn;
    }

    static RNTN trainModel(DataSetIterator dataIter, Word2Vec vec, RNTN model){
//      train model
        while(dataIter.hasNext()) {
        // this is looped with fit
            List<Tree> trees = vec.getTreesWithLabels(dataIter.nextSentence(), dataIter.currentLabel(), Arrays.asList("0", "1", "2", "3", "4"));
            model.fit(trees);
        }
        return model;
    }

    static void evalModel(List<Tree> testData, INDArray testLabels, RNTN model) {
        // RNTN evalu will eval per node - each sentence is a parse tree
        // rntn eval - positive and negative sentiment

        Evaluation eval = new Evaluation();
        List<INDArray> predictedOutput = model.output(testData);
        // Get labels
        eval.eval(predictedOutput, testLabels);
        log.info(eval.stats());
    }

    public void exec(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);

        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        log.info("Load & vectorize data....");
        DataSetIterator dataIter = loadData();
        Word2Vec vec = buildVectors();

        log.info("Build model....");
        RNTN model = buildModel(vec);

        log.info("Train model....");
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));
        trainModel(dataIter, vec, model);

        log.info("Evaluate model....");
        DataSetIterator testData = loadData();
        evalModel(testData, testLabels, model);

        log.info("****************Example finished********************");

    }

    public static void main(String[] args) throws Exception {

        new RNTNTwitterExample().exec(args);

        }

    }
}

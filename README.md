DL4J Examples 
=========================
*(based on version 0.0.3.3)*

Repository of Deeplearning4J neural net examples:

- Convolutional Neural Nets
- Deep-belief Neural Nets
- Glove Example
- Restricted Boltzmann Machines
- Recurrent Neural Nets
- Recursive Neural Nets
- Word2Vec Examples (includes TSNE)

---
## Development
We are in progress of developing and tuning these examples. If you notice issues, please log, and if you want to contribute, submit a pull request. Input is definitely welcome.

---
## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and [JavaDocs](http://deeplearning4j.org/doc/).

---
## Performance

| **Model Name**      | **Score** | **Status**   | **Training**  |
|---------------------|-----------|--------------|---------------|
| CNNIris             | F1 1.0    | Validate     | batch         |
| CNNMnist            | F1        | Not working  | batch         |
| CNNMnist2           | F1        | Not working  | batch         | 
| DBNCreateData       | F1 0.33   | Needs Tuning | batch         |          	
| DBNFullMnist        | F1 0.18   | Needs Tuning | full          |
| DBNIris             | F1 1.0    | Validate     | full          |
| DBNLWF              | F1 0.0    | Needs Tuning | batch         |
| DBNMnistRecontruct  | F1 0.017  | Needs Tuning | batch         |
| DBNSmallMnist       | F1 0.02   | Needs Tuning | full          |
| GloveRawSentence    | Sim 0     | Needs Tuning | batch         |
| MLPBackprop         | F1 0.16   | Needs Tuning | batch         |
| RBMCreateData	      | NA        | Validate     | full          |
| RBMIris             | NA        | Validate     | full          |
| RecurrentLSTMMnist  | NA        | Validate     | batch         |
| RecursiveAutoEncoder| NA        | Validate     | batch         |
| RNTNTwitter         | NA        | Develop      | batch         |
| TSNEBarnesHut       | NA        | Not working  | NA            |
| TSNEStandard        | NA        | Working      | NA            |
| Word2VecRawText     | Sim 0     | Needs Tuning | batch         |
    

CNN & DBN Iris examples are classifying only on 1 prediction with perfect score - need to verify
 MLP with backward true is showing all values and coming out with more realistic scores

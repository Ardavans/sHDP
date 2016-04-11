# sHDP: Nonparametric Topic Modeling with Word Vectors 

Implementation for the spherical HDP (sHDP) introduced in [*Nonparametric Spherical Topic Modeling with Word Embeddings*](http://arxiv.org/pdf/1604.00126v1.pdf). 

The code reads a dataset from the data folder and runs a nonparametric topic model with vMF likelihood. The generated topics are saved in the results folder. 

An example of the run command: 

```./runner.py -is 1 -alpha 1 -gamma 2 -Nmax 40 -kappa_sgd 0.6 -tau 0.8 -mbsize 10 -dataset nips ```

#Acknowledgement 
The code is developed based on [pyhsmm package](https://github.com/mattjj/pyhsmm).

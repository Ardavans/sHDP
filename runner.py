#!/usr/bin/env python

import argparse
from argparse import RawTextHelpFormatter
from collections import Counter
from core.core_distributions import  vonMisesFisherLogNormal
import csv
import sys
import pickle as pk
import numpy as np
from HDP import models
from HDP.util.general import sgd_passes
from HDP.util.text import progprint
import operator

project_path =  ''
results_path = project_path+'results/'

def HDPRunner(args):
    num_dim = 50
    infseed = args['infSeed'] #1
    K = args['Nmax']
    alpha = args['alpha'] #1
    gamma = args['gamma'] #2
    tau = args['tau']
    kappa_sgd = args['kappa_sgd']
    mbsize = args['mbsize']
    datasetname = args['dataset']

    results_file = results_path+datasetname+'/topics_infseed_' + str(infseed)+ '_K_'+ str(K) +\
                   '_alpha_' + str(alpha) + '_gamma_' + str(gamma) + '_kappa_sgd_' +\
                   str(kappa_sgd)+ '_tau_'+ str(tau)+ '_mbsize_' + str(mbsize) +'.txt'
    results_file_noncnt = results_path+datasetname+'/topics_noncnt_infseed_' + str(infseed)+ '_K_'+ str(K) +\
                   '_alpha_' + str(alpha) + '_gamma_' + str(gamma) + '_kappa_sgd_' +\
                   str(kappa_sgd)+ '_tau_'+ str(tau)+ '_mbsize_' + str(mbsize) +'.txt'




    ################# Data generatati
    temp_file = open(project_path+'data/'+datasetname+'/texts.pk', 'rb')
    texts = pk.load(temp_file)
    temp_file.close()

    print ('Loading the glove dict file....')
    csv.field_size_limit(sys.maxsize)
    vectors_file = open(project_path+'data/'+datasetname+'/wordvec.pk', 'rb')
    vectors_dict = pk.load(vectors_file)




    ########### Runner

    print ('Main runner ...')

    def glovize_data(list_of_texts):
        all_data = []
        for text in list_of_texts:
            temp_list = []
            for word in text:
                try:
                    temp_list.append((np.array(vectors_dict[word[0]]).astype(float), word[1]))
                except:
                    pass
            all_data.append(np.array(temp_list))
        return all_data

    def glovize_data_wo_count(list_of_texts):
        all_data = []
        all_avail_words = []
        for text in list_of_texts:
            temp_list = []
            temp_list_words = []
            for word in text:
                try:
                    temp_list.append((np.array(vectors_dict[word[0]]).astype(float), word[1]))
                    temp_list_words.append(word[0])
                except:
                    pass
            all_data.append(np.array(temp_list))
            all_avail_words.append(np.array(temp_list_words))
        return all_data, all_avail_words





    temp1 = glovize_data(texts)
    temp2 = glovize_data_wo_count(texts)[0]
    #python 2 just zip
    #temp2 = zip(temp2, range(len(temp2)))
    '''
    Original SHDP for Python2 just uses zip(), 
    but zip() cannot be indexed in Python3.
    By Changing zip() to list(zip()), the code works for Py2 & 3 :)
    It works in python 3, but the code fails later on.... 
    '''
    temp2 = list(zip(temp2, range(len(temp2))))
    real_data = temp2[:]
    num_docs = len(real_data)
    print ('num_docs: ' +  str(num_docs))
    temp_words = glovize_data_wo_count(texts)[1]
    temp_words = temp_words[:num_docs]
    vocabulary = np.unique([j for i in temp_words for j in i])


    training_size = num_docs
    all_words = []
    for d in temp1:
        for w in d:
            all_words.append(w[0])


    np.random.seed(infseed)


    d = np.random.rand(num_dim,)
    d = d/np.linalg.norm(d)
    obs_hypparams = dict(mu_0=d,C_0=1,m_0=2,sigma_0=0.25)
    components=[vonMisesFisherLogNormal(**obs_hypparams) for itr in range(K)]



    HDP = models.HDP(alpha=alpha, gamma=gamma, obs_distns=components, num_docs=num_docs + 1)




    sgdseq = sgd_passes(tau=tau,kappa=kappa_sgd,datalist=real_data,minibatchsize=mbsize, npasses= 1)
    for t, (data, rho_t) in progprint(enumerate(sgdseq)):
        HDP.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(training_size), rho_t)





    ############# Add data and do mean field

    ###count based topics
    all_topics_pred = []
    all_topics_unique = []
    for i in range(num_docs):
        HDP.add_data(np.atleast_2d(real_data[i][0].squeeze()), i)
        HDP.states_list[-1].meanfieldupdate()
        predictions = np.argmax(HDP.states_list[-1].all_expected_stats[0],1)
        all_topics_pred.append(predictions)
        all_topics_unique.extend(np.unique(predictions))

    unique_topics = np.unique(all_topics_unique)
    topics_dict = {}
    for j in unique_topics:
        topics_dict[j] = []
    for k in range(num_docs):
        for kk in range(len(all_topics_pred[k])):
            topics_dict[all_topics_pred[k][kk]].append(temp_words[k][kk])

    for t in unique_topics:
        topics_dict[t] = Counter(topics_dict[t]).most_common(30)
        print(topics_dict[t])

    #now there is a dictionary
    topic_file = open(results_file, 'wb')
    for t in unique_topics:
        if len(topics_dict[t]) > 5:
            top_ordered_words = topics_dict[t][:20]
            #print top_ordered_words
            #'str' does not support the buffer interface py34
            str_to_write = (' '.join([i[0] for i in top_ordered_words]))
            topic_file.write(str_to_write.encode('UTF-8'))
            topic_file.write('\n'.encode('UTF-8'))
            # topic_file.write(' '.join([i[0] for i in top_ordered_words]))
            # topic_file.write('\n')
    topic_file.close()

    ###prob based topics
    topics_dict = {}
    for j in range(K):
        topics_dict[j] = {}
        for k in vocabulary:
            topics_dict[j][k] = 0

    for idx, doc in enumerate(temp_words):
        HDP.add_data(np.atleast_2d(real_data[idx][0].squeeze()), idx)
        HDP.states_list[-1].meanfieldupdate()
        temp_exp = HDP.states_list[-1].all_expected_stats[0]
        for idw, word in enumerate(doc):
            for t in range(K):
                topics_dict[t][word] += temp_exp[idw, t]

    sorted_topics_dict = []
    print('################################')
    for t in range(K):
        sorted_topics_dict.append(sorted(topics_dict[t].items(), key=operator.itemgetter(1), reverse=True)[:20])
        print (sorted_topics_dict[-1])

    topic_file = open(results_file_noncnt, 'wb')
    for t in range(K):
        if len(sorted_topics_dict[t]) > 5:
            top_ordered_words = sorted_topics_dict[t][:20] #
            str_to_write = (' '.join([i[0] for i in top_ordered_words]))
            topic_file.write(str_to_write.encode('UTF-8')) #python 3 will only write bytes; py2 writes str and bytes
            topic_file.write('\n'.encode('UTF-8'))
    topic_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This program runs sHDP on a prepared corpus.
    sample argument setting is as follows:
    python runner.py -is 1 -alpha 1 -gamma 2 -Nmax 40 -kappa_sgd 0.6 -tau 0.8 -mbsize 10 -dataset nips
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-is', '--infSeed', help='Seed for running the model', type = np.int32, required=True)
    parser.add_argument('-alpha', '--alpha', help='alpha hyperparameter for the low level stick breaking process', type = np.float,
                        required=True)
    parser.add_argument('-gamma', '--gamma', help='gamma hyperparameter for the top level stick breaking process',type = np.float, required=True)
    parser.add_argument('-Nmax', '--Nmax', help='maximum number of states',
                        type = np.int32 ,required=True)
    parser.add_argument('-kappa_sgd', '--kappa_sgd', help='kappa for SGD', type = np.float, required=True)
    parser.add_argument('-tau', '--tau', help='tau for SGD', type = np.float, required=True)
    parser.add_argument('-mbsize', '--mbsize', help='mbsize for SGD', type = np.float, required=True)
    parser.add_argument('-dataset', '--dataset', help='choose one of nips 20news wiki', required=True)
    args = vars(parser.parse_args())
    print(args)
    HDPRunner(args)
























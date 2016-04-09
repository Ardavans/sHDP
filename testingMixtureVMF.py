# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:30:37 2016

@author: kayhan
"""
#%%  ---- imports
import os
import sys

print "Hello ..."

sys.path.append('/Users/kayhan/Projects/clean_sHDP/')

from core import  models
from core import distributions
#from pyhsmm.util.general import sgd_passes
#from pyhsmm.util.text import progprint_xrange, progprint


import numpy as np
np.random.seed(1000)
#import seaborn as sns 
from matplotlib.pyplot import *

from core.util.stats import sample_vMF

#%pylab inline


#%% -----

d = np.random.rand(2,)
d = d/np.linalg.norm(d)
#kappa = 100


figure(figsize=[15,4])
kappa = 1
samples = sample_vMF(d,kappa,size=100)
subplot(1,3,1)
scatter(samples[:,0], samples[:,1])
plot(d[0],d[1],'.r')
axis([-1.5,1.5,-1.5,1.5])

kappa = 10
samples = sample_vMF(d,kappa,size=100)
subplot(1,3,2)
scatter(samples[:,0], samples[:,1])
plot(d[0],d[1],'.r')
axis([-1.5,1.5,-1.5,1.5])


kappa = 100
samples = sample_vMF(d,kappa,size=100)
subplot(1,3,3)
scatter(samples[:,0], samples[:,1])
plot(d[0],d[1],'.r')
axis([-1.5,1.5,-1.5,1.5])


assert abs((samples**2).sum(axis=1) - 1).sum()/samples.shape[0] < 1e-5 , "samples do not live on the sphere !!"




#%% --- test initializing the class


reload(distributions)
vmf = distributions.vonMisesFisherLogNormal(mu_0=d,C_0=1,m_0=2,sigma_0=0.25)

data = vmf.rvs(100)
scatter(data[:,0], data[:,1])
plot(d[0],d[1],'.r')
_ = axis([-1.5,1.5,-1.5,1.5])


#%% --- test mixture distribution
# hyperparameters
alpha_0=5.0
obs_hypparams = dict(mu_0=d,C_0=0.1,m_0=3,sigma_0=0.25)

# create the model
priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.vonMisesFisherLogNormal(**obs_hypparams) for itr in range(3)])

# generate some data
#data = priormodel.rvs(200)
data = np.load('data/simpleMixture.npy')


print "weight : ", priormodel.weights.weights


#del priormodel

figure()
plot(data[:,0],data[:,1],'kx')
_ = axis([-1.5,1.5,-1.5,1.5])
title('data')


#%% --- build the model and make inference
reload(distributions)

obs_hypparams = dict(mu_0=d,C_0=1,m_0=2,sigma_0=0.25)
posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.vonMisesFisherLogNormal(**obs_hypparams) for itr in range(5)])

posteriormodel.add_data(data)

import copy
from core.util.text import progprint_xrange

allscores = [] # variational lower bounds on the marginal data log likelihood
allmodels = []
for superitr in range(5):
    # Gibbs sampling to wander around the posterior
    print 'Gibbs Sampling'
    for itr in progprint_xrange(100):
        posteriormodel.resample_model()

    # mean field to lock onto a mode
    print 'Mean Field'
    scores = [posteriormodel.meanfield_coordinate_descent_step()
                for itr in progprint_xrange(100)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(posteriormodel))

#%% -----
import operator
models_and_scores = sorted([(m,s[-1]) for m,s
    in zip(allmodels,allscores)],key=operator.itemgetter(1),reverse=True)


figure()
for scores in allscores:
    plot(scores)
title('model vlb scores vs iteration')


#%% -------
#from cycler import cycler

#plt.rc('lines', linewidth=4)
#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
#                           cycler('linestyle', ['-', '--', ':', '-.'])))

colorDict = {0:'r',1:'g',2:'b',3:'c',4:'m'}
fcn = lambda x:colorDict[x]    
                       
figure()
m = models_and_scores[0][0]
l = m.labels_list[0]
scatter(data[:,0], data[:,1],
        edgecolor = np.array( map(fcn, l.r.argmax(axis=1))),
        facecolor = np.array( map(fcn, l.r.argmax(axis=1)))  )
_ = axis([-1.5,1.5,-1.5,1.5])

for c in m.components:
    plot(c.mu_mf[0], c.mu_mf[1],'*r',markersize=15)
    

show()

#%% --------
# test sdg
# scores = []
# posteriormodel2 = models.Mixture(alpha_0=alpha_0,
#         components=[distributions.vonMisesFisherLogNormal(**obs_hypparams) for itr in range(5)])
#
# sgdseq = sgd_passes(tau=0.1, kappa=0.7, datalist=data, minibatchsize=10, npasses= 50)
# for t, (data, rho_t) in progprint(enumerate(sgdseq)):
#     #print 'rho_t', rho_t
#
#     #print (data[0])
#     posteriormodel2.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(200), rho_t)
#     if (t + 1) % (num_docs /mbsize) == 0:
#         scores.append(posteriormodel2.log_likelihood([i[0] for i in data[-50:]], [i[1] for i in data[-50:]]))
#         print 'score: ', scores[-1]
#

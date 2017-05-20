from __future__ import division
import numpy as np
np.seterr(invalid='raise')
import copy

from scipy.special import digamma

from HDP.basic.distributions import Multinomial
from HDP.util.general import cumsum, rcumsum



################# HDP dishes matrix

class _HDPMatrixBase(object):
    def __init__(self,num_states=None, num_docs = None, alpha=None,alphav=None,trans_matrix=None):
        self.N = num_states
        self.ND = num_docs
        if trans_matrix is not None:
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav,
                weights=row) for row in trans_matrix]
        elif None not in (alpha,self.N) or (alphav is not None and alphav.ndim <2):
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav)
                    for n in range(num_docs)] # sample from prior
        elif None not in (alpha,self.N) or (alphav is not None and alphav.ndim == 2):
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav[n,:])
                    #python 2 xrange
                    #for n in xrange(self.N)] # sample from prior
                    #python 3
                    for n in range(self.N)] # sample from prior 

    @property
    def trans_matrix(self):
        return np.array([d.weights for d in self._row_distns])

    @trans_matrix.setter
    def trans_matrix(self,trans_matrix):
        N = self.N = trans_matrix.shape[1]
        if self.alphav.ndim < 2:
            self._row_distns = \
                    [Multinomial(alpha_0=self.alpha,K=N,alphav_0=self.alphav,weights=row)
                            for row in trans_matrix]

    @property
    def alpha(self):
        return self._row_distns[0].alpha_0

    @alpha.setter
    def alpha(self,val):
        for distn in self._row_distns:
            distn.alpha_0 = val

    @property
    def alphav(self):
        return self._row_distns[0].alphav_0

    @alphav.setter
    def alphav(self,weights):
        if weights.ndim < 2:
            for distn in self._row_distns:
                distn.alphav_0 = weights

    def copy_sample(self):
        new = copy.copy(self)
        new._row_distns = [distn.copy_sample() for distn in self._row_distns]
        return new

class _HDPMatrixMeanField(_HDPMatrixBase):
    @property
    def exp_expected_log_trans_matrix(self):
        return np.exp(np.array([distn.expected_log_likelihood()
            for distn in self._row_distns]))

    def meanfieldupdate(self,expected_states_doc_num_pair):
        assert isinstance(expected_states_doc_num_pair,list) and len(expected_states_doc_num_pair) > 0
        #trans_softcounts = sum(expected_states)

        # for distn, counts in zip(self._row_distns,trans_softcounts):
        #     distn.meanfieldupdate(None,counts)
        for expcnt in expected_states_doc_num_pair:
                self._row_distns[expcnt[1]].meanfieldupdate(None,expcnt[0])
        return self

    def get_vlb(self):
        return sum(distn.get_vlb() for distn in self._row_distns)

    def _resample_from_mf(self):
        for d in self._row_distns:
            d._resample_from_mf()

class _HDPMatrixSVI(_HDPMatrixMeanField):
    def meanfield_sgdstep(self,expected_states_doc_num_pair,minibatchfrac,stepsize):
        assert isinstance(expected_states_doc_num_pair,list)
        if len(expected_states_doc_num_pair) > 0:
            #trans_softcounts = sum(expected_transcounts)
            for expcnt in expected_states_doc_num_pair:
                self._row_distns[expcnt[1]].meanfield_sgdstep(None,expcnt[0],minibatchfrac,stepsize)
        return self

class _DATruncHDPBase(_HDPMatrixBase):
    # NOTE: self.beta stores \beta_{1:K}, so \beta_{\text{rest}} is implicit

    def __init__(self,gamma,alpha,num_states,num_docs, beta=None,trans_matrix=None):
        self.N = num_states
        self.ND = num_docs
        self.gamma = gamma
        self._alpha = alpha
        if beta is None:
            # beta = np.ones(num_states) / (num_states + 1)
            beta = self._sample_GEM(gamma,num_states)
        assert not np.isnan(beta).any()

        betafull = np.concatenate(((beta,(1.-beta.sum(),))))

        super(_DATruncHDPBase,self).__init__(
                num_states=self.N, num_docs = self.ND, alphav=alpha*betafull,trans_matrix=trans_matrix)

        self.beta = beta

    @staticmethod
    def _sample_GEM(gamma,K):
        v = np.random.beta(1.,gamma,size=K)
        return v * np.concatenate(((1.,),np.cumprod(1.-v[:-1])))

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self,beta):
        self._beta = beta
        self.alphav = self._alpha * np.concatenate((beta,(1.-beta.sum(),)))

    @property
    def exp_expected_log_trans_matrix(self):
        return super(_DATruncHDPBase,self).exp_expected_log_trans_matrix[:,:-1].copy()

    @property
    def trans_matrix(self):
        return super(_DATruncHDPBase,self).trans_matrix[:,:-1].copy()


class _DATruncHDPSVI(_DATruncHDPBase,_HDPMatrixSVI, _HDPMatrixMeanField):
    def meanfieldupdate(self,expected_transcounts):
        super(_DATruncHDPSVI,self).meanfieldupdate(
                self._pad_zeros(expected_transcounts))

    def meanfield_sgdstep(self,expected_states_doc_num_pair,minibatchfrac,stepsize):
        # NOTE: since we take a step on q(beta) and on q(pi) at the same time
        # (as usual with SVI), we compute the beta gradient and perform the pi
        # step before applying the beta gradient

        beta_gradient = self._beta_gradient()
        super(_DATruncHDPSVI,self).meanfield_sgdstep(
                self._pad_zeros(expected_states_doc_num_pair),minibatchfrac,stepsize) #TODO make sure you don't need self._pad_zeros()
        self.beta = self._feasible_step(self.beta,beta_gradient,stepsize)
        #print self.beta

        #print self._row_distns[0]._alpha_mf - self._row_distns[1]._alpha_mf
        assert (self.beta >= 0.).all() and self.beta.sum() < 1
        return self

    def _pad_zeros(self,counts):
        if isinstance(counts,tuple):
            return (np.pad(counts[0],((0,1)),mode='constant',constant_values=0), counts[1])
        return [self._pad_zeros(c) for c in counts]

    @staticmethod
    def _feasible_step(pt,grad,stepsize):
        def newpt(pt,grad,stepsize):
            return pt + stepsize*grad
        def feas(pt):
            return (pt>0.).all() and pt.sum() < 1. and not np.isinf(1./(1-cumsum(pt))).any()
        grad = grad / np.abs(grad).max()
        while True:
            new = newpt(pt,grad,stepsize)
            if feas(new):
                return new
            else:
                grad /= 1.5

    def _beta_gradient(self):
        if not isinstance(self._alpha, (np.ndarray, np.generic) ):
            return self._grad_log_p_beta(self.beta,self.gamma) + \
                sum(self._grad_E_log_p_pi_given_beta(self.beta, self._alpha,
                    distn._alpha_mf) for distn in self._row_distns)
        else:
            return self._grad_log_p_beta(self.beta,self.gamma) + \
                sum(self._grad_E_log_p_pi_given_beta(self.beta, self._alpha[idx,:-1],
                    distn._alpha_mf) for idx, distn in enumerate(self._row_distns))

    @staticmethod
    def _grad_log_p_beta(beta,alpha):
        # NOTE: switched argument name gamma <-> alpha
        return  -(alpha-1)*rcumsum(1./(1-cumsum(beta))) \
                + 2*rcumsum(1./(1-cumsum(beta,strict=True)),strict=True)

    def _grad_E_log_p_pi_given_beta(self,beta,gamma,alphatildes):
        # NOTE: switched argument name gamma <-> alpha
        retval = gamma*(digamma(alphatildes[:-1]) - digamma(alphatildes[-1])) \
                - gamma * (digamma(gamma*beta) - digamma(gamma))
        return retval

    def get_vlb(self):
        return super(_DATruncHDPSVI,self).get_vlb() \
                + self._beta_vlb()

    def _beta_vlb(self):
        return np.log(self.beta).sum() + self.gamma*np.log(1-cumsum(self.beta)).sum() \
               - 3*np.log(1-cumsum(self.beta,strict=True)).sum()


class DATruncHDP(_DATruncHDPSVI):
    pass





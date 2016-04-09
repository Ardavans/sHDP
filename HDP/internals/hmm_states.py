from __future__ import division
import numpy as np
from numpy import newaxis as na
import abc
import copy

from HDP.util.stats import sample_discrete
try:
    from HDP.util.cstats import sample_markov
except ImportError:
    from HDP.util.stats import sample_markov
from HDP.util.general import rle

################ HDP model

class _HDPStatesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,model,T=None, doc_num = None, data=None,stateseq=None, num_states = None,
            generate=True,initialize_from_prior=True):
        self.model = model

        # self.num_states = num_states
        self.T = T if T is not None else data.shape[0] #num of words
        self.data = data
        self.doc_num = doc_num

        self.clear_caches()

        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new

    _kwargs = {}  # used in subclasses for joblib stuff

    ### model properties

    @property
    def obs_distns(self):
        return self.model.obs_distns

    @property
    def trans_matrix(self):
        return self.model.trans_distn.trans_matrix


    @property
    def num_states(self):
        return self.model.num_states

    ### convenience properties

    @property
    def stateseq_norep(self):
        return rle(self.stateseq)[0]


    ### generation

    @abc.abstractmethod
    def generate_states(self):
        pass

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._normalizer = None

    @property
    def aBl(self):
        if self._aBl is None : #or self._aBl is not None:
            data = self.data

            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    @abc.abstractmethod
    def log_likelihood(self):
        pass


class HDPStates(_HDPStatesBase):
    ### generation

    def generate_states(self):
        T = self.T
        doc_num = self.doc_num
        state_distn = self.trans_matrix[doc_num, :]

        stateseq = np.zeros(T,dtype=np.int32)
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(state_distn)

        self.stateseq = stateseq
        return stateseq


    def log_likelihood(self):
        if self._normalizer is None:
            self.messages_forwards_log()
        return self._normalizer



    @staticmethod
    def _messages_forwards_log(trans_matrix_docnum, log_likelihoods):
        errs = np.seterr(over='ignore')
        aBl = log_likelihoods
        np.seterr(**errs)
        return aBl + trans_matrix_docnum

    def messages_forwards_log(self):
        alphal = self._messages_forwards_log(self.trans_matrix[self.doc_num, :], self.mf_aBl)
        assert not np.any(np.isnan(alphal))
        self._normalizer = np.logaddexp.reduce(np.logaddexp.reduce(alphal, 1))
        return alphal



    ### Gibbs sampling

    def resample_log(self):
        #betal = self.messages_backwards_log()
        self.sample_forwards_log()


    def resample(self):
        return self.resample_log()

    @staticmethod
    def _sample_forwards_log(trans_matrix_docnum, log_likelihoods):
        A = trans_matrix_docnum
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = trans_matrix_docnum
        for idx in xrange(T):
            logdomain = aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            if np.any(np.isfinite(logdomain)):
                stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            else:
                stateseq[idx] = sample_discrete(nextstate_unsmoothed)
            nextstate_unsmoothed = trans_matrix_docnum

        return stateseq

    def sample_forwards_log(self,betal):
        self.stateseq = self._sample_forwards_log(self.trans_matrix[self.doc_num],self.aBl)


    ### Mean Field

    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((T,self.num_states))

            for idx, o in enumerate(self.obs_distns):
                #aBl[:,idx] = o.expected_log_likelihood(self.data).ravel()
                aBl[:,idx] = o.expected_log_likelihood([i[0] for i in self.data]).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._mf_aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix
        # return np.maximum(self.model.trans_distn.exp_expected_log_trans_matrix,1e-5)



    @property
    def all_expected_stats(self):
        return self.expected_states, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self._normalizer = vals
        self.stateseq = self.expected_states.argmax(1).astype('int32') # for plotting

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_matrix[self.doc_num,:],self.mf_aBl)

    def get_vlb(self):
        if self._normalizer is None:
            self.meanfieldupdate() # NOTE: sets self._normalizer
        return self._normalizer

    def _expected_statistics(self,trans_potential_docnum,likelihood_log_potential):
        alphal = self._messages_forwards_log(trans_potential_docnum, likelihood_log_potential)

        expected_states, normalizer = \
                self._expected_statistics_from_messages_slow(alphal)
        assert not np.isinf(expected_states).any()
        return expected_states, normalizer

    @staticmethod
    def _expected_statistics_from_messages_slow(alphal):
        expected_states = alphal
        expected_states -= expected_states.max(1)[:,na]
        np.exp(expected_states,out=expected_states)
        expected_states /= expected_states.sum(1)[:,na]
        normalizer = np.logaddexp.reduce(np.logaddexp.reduce(alphal, 1)) #TODO check this
        return expected_states, normalizer










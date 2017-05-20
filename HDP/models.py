from __future__ import division
import numpy as np
from numpy import newaxis as na

from core.core_abstractions import Model, ModelMeanField, ModelMeanFieldSVI
import os, sys
scriptpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'internals'))
sys.path.append(scriptpath)
#from internals import hmm_states, transitions
import hmm_states, transitions



class _HDPBase(Model):
    _states_class = hmm_states.HDPStates
    _trans_class = transitions.DATruncHDP

    def __init__(self,
            obs_distns,
            num_docs = None,
            trans_distn=None,
            alpha=None,alpha_a_0=None,alpha_b_0=None,trans_matrix=None, gamma=None):
        self.obs_distns = obs_distns
        self.states_list = []
        self.num_states = len(obs_distns)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        else:
            self.trans_distn = self._trans_class(
                    num_states=self.num_states,num_docs=num_docs, gamma=gamma, alpha=alpha,trans_matrix=trans_matrix)

        self._clear_caches()

    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()

    def add_data(self,data,doc_num,stateseq=None,**kwargs):
        self.states_list.append(
                self._states_class(
                    model=self,data=data,doc_num=doc_num, num_states =self.num_states,
                    stateseq=stateseq,**kwargs))

    def generate(self, T, doc_num, keep=True):
        s = self._states_class(model=self,T=T,doc_num = doc_num, initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq, s

    def _generate_obs(self,s):
        if s.data is None:
            # generating brand new data sequence
            s.data = [s.obs_distns[state].rvs() for idx, state in enumerate(s.stateseq)]
        else:
            # filling in missing data
            data = s.data
            nan_idx, = np.where(np.isnan(data).any(1))
            counts = np.bincount(s.stateseq[nan_idx],minlength=self.num_states)
            obs = [iter(o.rvs(count)) for o, count in zip(s.obs_distns,counts)]
            for idx, state in zip(nan_idx, s.stateseq[nan_idx]):
                data[idx] = obs[state].next()

        return s.data

    def log_likelihood(self,data=None, doc_num = None, **kwargs):
        if data is not None:
            if isinstance(data,np.ndarray):
                self.add_data(data=data, doc_num=doc_num, generate=False,**kwargs)
                return self.states_list.pop().log_likelihood()
            else:
                assert isinstance(data,list) and isinstance(doc_num,list)
                loglike = 0.
                for idx, d in enumerate(data):
                    self.add_data(data=d, doc_num=doc_num[idx], generate=False,**kwargs)
                    #self._clear_caches()
                    loglike += self.states_list.pop().log_likelihood()
                return loglike
        else:
            return sum(s.log_likelihood() for s in self.states_list)

    @property
    def stateseqs(self):
        return [s.stateseq for s in self.states_list]




class _HDPMeanField(_HDPBase,ModelMeanField):
    def meanfield_coordinate_descent_step(self,num_procs=0):
        self._meanfield_update_sweep(num_procs=num_procs)
        #return self._vlb()

    def _meanfield_update_sweep(self,num_procs=0):
        # NOTE: we want to update the states factor last to make the VLB
        # computation efficient, but to update the parameters first we have to
        # ensure everything in states_list has expected statistics computed
        self._meanfield_update_states_list(
            [s for s in self.states_list if not hasattr(s,'expected_states')],
            num_procs)

        self.meanfield_update_parameters()
        self.meanfield_update_states(num_procs)

    def meanfield_update_parameters(self):
        self.meanfield_update_obs_distns()
        self.meanfield_update_trans_distn()

    def meanfield_update_obs_distns(self):
        for state, o in enumerate(self.obs_distns):
            o.meanfieldupdate([np.array([i[0] for i in s.data]) for s in self.states_list],
                    [[s.expected_states[:,state] * np.array([i[1] for i in s.data])][0] for s in self.states_list])


    def meanfield_update_trans_distn(self):
        self.trans_distn.meanfieldupdate(
                [([s.expected_states * np.array([i[1] for i in s.data])[:, na]][0], s.doc_num) for s in self.states_list])


    def meanfield_update_states(self,num_procs=0):
        self._meanfield_update_states_list(self.states_list,num_procs=num_procs)

    def _meanfield_update_states_list(self,states_list,num_procs=0):
        if num_procs == 0:
            for s in states_list:
                s.meanfieldupdate()



class _HDPSVI(_HDPBase,ModelMeanFieldSVI):
    # NOTE: classes with this mixin should also have the _HMMMeanField mixin for
    # joblib/multiprocessing stuff to work
    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,num_procs=0,**kwargs):
        ## compute the local mean field step for the minibatch
        mb_states_list = self._get_mb_states_list(minibatch,**kwargs)
        if num_procs == 0:
            for s in mb_states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(mb_states_list,num_procs)

        ## take a global step on the parameters
        self._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)
        print ("")
    def _get_mb_states_list(self,minibatch,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_states_list = []
        for mb in minibatch: #minibatch is a pair of word sequence and doc index
            self.add_data(mb[0], mb[1],generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        self._meanfield_sgdstep_obs_distns(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_trans_distn(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, o in enumerate(self.obs_distns):
            o.meanfield_sgdstep(
                    [np.array([i[0] for i in s.data]) for s in mb_states_list],
                    [[s.expected_states[:,state] * np.array([i[1] for i in s.data])][0] for s in mb_states_list],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.trans_distn.meanfield_sgdstep(
                [([s.expected_states * np.array([i[1] for i in s.data])[:, na]][0], s.doc_num) for s in mb_states_list],
                minibatchfrac,stepsize)


class HDP(_HDPSVI, _HDPMeanField, _HDPBase):
    pass
















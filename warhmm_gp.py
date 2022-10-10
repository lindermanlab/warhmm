import numpy as np
import numpy.random as npr
import scipy.stats
from scipy import linalg as sclin
import torch
from tqdm.auto import trange
from torch.distributions import MultivariateNormal
import pickle
import os
from util import random_rotation, sum_tuples, kron_A_N
import wandb
import time
from numba import njit, prange
from twarhmm import TWARHMM, LinearRegressionObservations, Posterior

device = torch.device('cpu')
dtype = torch.float64
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()
kernel_ridge = 1e-4


class TWARHMM_GP(TWARHMM):
    def __init__(self, config, taus, kernel):
        super().__init__(config, taus)  # config is a dictionary containing parameters
        self.taus = taus
        self.observations = LinearRegressionObservations_GP(self.num_discrete_states, self.data_dim,
                                                        self.covariates_dim, self.taus, kernel,
                                                        config["covariance_reg"], random_weights=False)

    def fit(self, train_dataset, test_dataset, seed=0, num_epochs=50, fit_observations=True, fit_transitions=True, fit_tau=False, fit_kernel_params=True):
        # Fit using full batch EM
        num_train = sum([len(data["data"]) for data in train_dataset])
        num_test = sum([len(data["data"]) for data in test_dataset])
        # Initialize with a random posterior
        total_states = self.num_discrete_states*self.observations.num_taus
        posteriors = [Posterior(self, data_dict, total_states) for data_dict in train_dataset]
        for posterior in posteriors:
            posterior.update()
        continuous_expectations, discrete_expectations = self.compute_expected_suff_stats(train_dataset, posteriors, self.taus, fit_observations, fit_transitions)
        train_lls = []
        test_lls = []

        # Main loop
        for itr in trange(num_epochs):
            #print(itr)
            self.M_step(continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau, fit_kernel_params)

            for posterior in posteriors:
                posterior.update()

            # Compute the expected sufficient statistics under the new posteriors
            continuous_expectations, discrete_expectations = self.compute_expected_suff_stats(train_dataset, posteriors, self.taus, fit_observations, fit_transitions)

            # Store the average train likelihood
            avg_train_ll = (sum([p.marginal_likelihood() for p in posteriors]) + self.observations.log_prior_likelihood().detach().numpy())/ num_train
            train_lls.append(avg_train_ll) # TO DO: need to add prior log likelihood to overall objective function

            # Compute the posteriors for the test dataset too
            test_posteriors = [Posterior(self,data_dict,total_states) for data_dict in test_dataset]

            for posterior in test_posteriors:
                posterior.update()

            # Store the average test likelihood
            avg_test_ll = (sum([p.marginal_likelihood() for p in test_posteriors]) ) / num_test
            test_lls.append(avg_test_ll)

        # convert lls to arrays
        train_lls = np.array(train_lls)
        test_lls = np.array(test_lls)
        return train_lls, test_lls, posteriors, test_posteriors

    def fit_stoch(self, train_dataset, test_dataset, forgetting_rate=-0.5, seed=0, num_epochs=5, fit_observations=True,
                    fit_transitions=True, fit_tau = True, compute_posteriors=True, fit_kernel_params=True, wandb_log=False):
        # Get some constants
        num_batches = len(train_dataset)
        taus = np.array(self.taus)
        num_test = sum([len(data["data"]) for data in test_dataset])
        total_states = self.num_discrete_states * len(self.taus)
        num_train = sum([len(data["data"]) for data in train_dataset])

        # Initialize the step size schedule
        schedule = np.arange(1, 1 + num_batches * num_epochs) ** (forgetting_rate)

        # Initialize progress bars
        outer_pbar = trange(num_epochs)
        inner_pbar = trange(num_batches)
        outer_pbar.set_description("Epoch")
        inner_pbar.set_description("Batch")

        # Main loop
        rng = npr.RandomState(seed)
        train_lls = []
        test_lls = []

        it_times = np.zeros((num_epochs,num_batches))

        for epoch in range(num_epochs):
            perm = rng.permutation(num_batches)

            inner_pbar.reset()
            for itr in range(num_batches):
                t = time.time()
                minibatch = [train_dataset[perm[itr]]]
                this_num_train = len(minibatch[0]["data"])

                posteriors = [Posterior(self, data, total_states) for data in minibatch]

                # E step: on this minibatch
                for posterior in posteriors:
                    posterior.update()

                if itr == 0 and epoch == 0: continuous_expectations, discrete_expectations = self.compute_expected_suff_stats(
                    minibatch, posteriors, taus, fit_observations, fit_transitions)
                # M step: using current stats
                self.M_step(continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau, fit_kernel_params)

                these_continuous_expectations, these_discrete_expectations = self.compute_expected_suff_stats(minibatch,
                                                                                                              posteriors,
                                                                                                              taus, fit_observations,
                                                                                                              fit_transitions)
                rescale = lambda x: num_train / this_num_train * x

                # Rescale the statistics as if they came from the whole dataset
                rescaled_cont_stats = tuple(rescale(st) for st in these_continuous_expectations)
                rescaled_disc_stats = tuple(rescale(st) for st in these_discrete_expectations)

                # Take a convex combination of the statistics using current step sz
                stepsize = schedule[epoch * num_batches + itr]
                continuous_expectations = tuple(
                    sum(x) for x in zip(tuple(st * (1 - stepsize) for st in continuous_expectations),
                                        tuple(st * (stepsize) for st in rescaled_cont_stats)))
                discrete_expectations = tuple(
                    sum(x) for x in zip(tuple(st * (1 - stepsize) for st in discrete_expectations),
                                        tuple(st * (stepsize) for st in rescaled_disc_stats)))

                # Store the normalized log likelihood for this minibatch
                avg_mll = (sum([p.marginal_likelihood() for p in posteriors])+ self.observations.log_prior_likelihood().detach().numpy()) / this_num_train
                train_lls.append(avg_mll)

                elapsed = time.time()-t
                #print(elapsed)
                it_times[epoch,itr] = elapsed
                inner_pbar.set_description("Batch LL: {:.3f}".format(avg_mll))
                inner_pbar.update()
                if wandb_log: wandb.log({'batch_ll': avg_mll})

            # Evaluate the likelihood and posteriors on the test dataset
            if compute_posteriors:
                test_posteriors = [Posterior(self, test_data, total_states, seed) for test_data in test_dataset]
                for posterior in test_posteriors:
                    posterior.update()
                avg_test_mll = (sum([p.marginal_likelihood() for p in test_posteriors])) / num_test
            else:
                mlls = []
                for test_data in test_dataset:
                    posterior = Posterior(self, test_data, total_states, seed)
                    posterior.update()
                    mlls.append(posterior.marginal_likelihood())
                avg_test_mll = np.sum(mlls)/ num_test
                test_posteriors = None
            test_lls.append(avg_test_mll)
            outer_pbar.set_description("Test LL: {:.3f}".format(avg_test_mll))
            outer_pbar.update()
            if wandb_log: wandb.log({'test_ll': avg_test_mll})


        # convert lls to arrays
        train_lls = np.array(train_lls)
        test_lls = np.array(test_lls)

        print('average iteration time: ', it_times.mean())
        return train_lls, test_lls, posteriors, test_posteriors

    def M_step(self, continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau,
               fit_kernel_params, hyper_M_iter=100):
        if fit_transitions: self.transitions.M_step(discrete_expectations, fit_tau=fit_tau)
        if fit_observations: self.observations.M_step(continuous_expectations)
        if fit_kernel_params: self.observations.hyper_M_step(niter=hyper_M_iter, learning_rate=1e-6)

    def compute_expected_suff_stats(self, dataset, posteriors, taus, fit_observations=True, fit_transitions=False):
        assert isinstance(dataset, list)
        assert isinstance(posteriors, list)

        # Helper function to compute expected counts and sufficient statistics
        # for a single time series and corresponding posterior.
        def _compute_expected_suff_stats(data, posterior, taus, fit_transitions):
            dxdxT, dxxT, xxT = data['suff_stats_gp']
            (fancy_e_z, fancy_e_t) = posterior.expected_transitions()

            T,D,_ = xxT.shape
            _,Dx,_ = dxdxT.shape
            M = self.observations.num_taus
            K = self.num_discrete_states

            w = posterior.expected_states().reshape((T,K,M))

            # initializing, in case fit_observations or fit_transitions is false
            fancy_e_z_over_T = np.zeros((self.num_discrete_states, self.num_discrete_states))
            fancy_e_t_over_T = np.zeros((len(self.taus), len(self.taus)))
            q_one = np.zeros((self.num_discrete_states, len(self.taus)))

            xxTw = np.zeros((self.num_discrete_states, D, D, len(self.taus)))
            dxxTw = np.zeros((self.num_discrete_states, Dx, D, len(self.taus)))
            dxdxTw = np.zeros((self.num_discrete_states, Dx, Dx, len(self.taus)))

            if fit_observations:

                xxTw = np.einsum('tij, tkm -> kijm', xxT, w, optimize='optimal') # K x D x D x M
                dxxTw = np.einsum('tij, tkm -> kijm', dxxT, w, optimize='optimal')
                dxdxTw = np.einsum('tij, tkm -> kijm', dxdxT, w, optimize='optimal')

                wk = w.sum(axis=(0,2))

            if fit_transitions:
                fancy_e_z_over_T = np.einsum('tij->ij', fancy_e_z, optimize='optimal')
                fancy_e_t_over_T = np.einsum('tij->ij', fancy_e_t, optimize='optimal')

                q_one = posterior.expected_states()[0]

            stats = (tuple((xxTw, dxxTw, dxdxTw, wk)),
                     tuple((fancy_e_z_over_T, fancy_e_t_over_T, q_one)))

            return stats

        # Sum the expected stats over the whole dataset
        stats = (None,None)
        for data, posterior in zip(dataset, posteriors):
            these_stats = _compute_expected_suff_stats(data, posterior, taus, fit_transitions)
            stats_cont = sum_tuples(stats[0], these_stats[0])
            stats_disc = sum_tuples(stats[1], these_stats[1])
            stats = (stats_cont, stats_disc)
        return stats


class LinearRegressionObservations_GP(LinearRegressionObservations):
    """
    Wrapper for a collection of Gaussian observation parameters.
    """

    def __init__(self, num_states, data_dim, covariate_dim, taus, kernel, covariance_reg, random_weights=True):
        super().__init__(num_states, data_dim, covariate_dim, taus, covariance_reg, random_weights=True)
        # self.priorCov = kernel(to_t(taus)) # covariance matrix  num_discrete_states x ndim(tau) x ndim(tau)
        self.num_taus = len(taus)
        self.weight_gp_kernel = kernel

        # changing shape of weights to match KxDxDxM
        if random_weights:
            self.weights = np.zeros((num_states, data_dim, covariate_dim, self.num_taus))
            for k in range(num_states):
                for m in range(self.num_taus):
                    self.weights[k, :, :data_dim, m] = scipy.linalg.logm(random_rotation(data_dim, theta=np.pi / 20))
        else:
            self.weights = np.zeros((num_states, data_dim, covariate_dim, self.num_taus))

        # adding in covs here to adjust initialization more easily
        self.covs = .15 * np.tile(np.eye(data_dim), (num_states, 1, 1))

    @staticmethod
    def precompute_suff_stats(dataset):
        """
        Compute the sufficient statistics of the linear regression for each
        data dictionary in the dataset. This modifies the dataset in place.

        Parameters
        ----------
        dataset: a list of data dictionaries.

        Returns
        -------
        Nothing, but the dataset is updated in place to have a new `suff_stats_gp`
            key, which contains a tuple of sufficient statistics.
        """
        # TODO: diff or dx??? leaning towards diff based on scott's derivation
        for data in dataset:
            x = data['data'] # t = 2 : T 
            # diff = np.diff(x, axis=0)
            phi = data['covariates'] # t = 1:T-1
            diff = x[1:] - x[:-1] # easier to read for now
            # TODO: update to generalize for lags >1
            if x.shape[1] == phi.shape[1]:  # no bias
                dx = x - phi
            else:
                dx = x - phi[:, :-1]
            data['suff_stats_gp'] = (np.einsum('ti,tj->tij', dx, dx), # dxn dxn.T
                                  np.einsum('ti,tj->tij', dx, phi), # dxn xn-1.T
                                  np.einsum('ti,tj->tij', phi, phi))

    def M_step(self, continuous_expectations):
        """
        Compute the linear regression parameters given the expected
        sufficient statistics.

        Note: add a little bit (1e-4 * I) to the diagonal of each covariance
            matrix to ensure that the result is positive definite.


        Parameters
        ----------
        stats: a tuple of expected sufficient statistics

        Returns
        -------
        Nothing, but self.weights and self.covs are updated in place.
        """
        # stats = tuple((dxxT_over_Etau,xxT_over_Etau))
        # H,wxxT = continuous_expectations # KxDxDxM, KxMxDxD
        # w,xxT = continuous_expectations

        xxTw,dxxTw, dxdxTw, wk = continuous_expectations # LD: modified this to try tensor update
        
        D = self.covariate_dim
        Dx = self.data_dim
        K = self.num_states
        M = len(self.taus)
        Qinv = np.linalg.inv(self.covs)

        Ker = self.weight_gp_kernel(to_t(self.taus)).detach().numpy() + kernel_ridge*np.eye(self.num_taus)[None,:,:] # add small ridge for stability
        Kinv = np.linalg.inv(Ker)

        # tensor version ... maybe slower but for debugging purposes
        Ahat = np.zeros((K, Dx, D, M))
        Qhat = np.zeros((K, Dx, Dx))


        for k in range(K):
            J1t = kron_A_N(Kinv[k,:,:], Dx*D)
            J2t = sclin.block_diag(*np.kron(xxTw[k].transpose(2,0,1), Qinv[k]))

            Sigma = J1t + J2t
            QinvdxxTw = np.einsum('pj, jlm -> plm', Qinv[k], dxxTw[k], optimize='optimal') # D x D x M

            mu = np.linalg.inv(Sigma) @ QinvdxxTw.flatten(order='F') # linear solve might be faster here
            Ahat[k,:,:,:] = mu.reshape(Dx,D,M,order='F') # C style reordering would be faster but using column vector convention for now

            # update covariance 
            AxxTwAT = np.einsum('ijm, jlm, plm -> ip', Ahat[k], xxTw[k], Ahat[k], optimize='optimal')
            dxxTATw = np.einsum('jlm, plm -> jp', dxxTw[k], Ahat[k], optimize='optimal')
            
            Qhat[k,:,:] = (AxxTwAT + dxdxTw[k].sum(axis=2) - dxxTATw - dxxTATw.T) / wk[k]

        # update stored parameters
        self.weights = Ahat
        self.covs = Qhat

    def log_likelihoods(self, data):
        """
        Compute the matrix of log likelihoods of data for each state.
        (I like to use torch.distributions for this, though it requires
         converting back and forth between numpy arrays and pytorch tensors.)
        Parameters
        ----------
        data: a dictionary with multiple keys, including "data", the TxD array
            of observations for this mouse.
        Returns
        -------
        log_likes: a TxK array of log likelihoods for each datapoint and
            discrete state.
        """
        y = to_t(data["data"])
        x = data["covariates"]

        # T,_ = x.shape
        #
        # K,Dx,D,M = self.weights.shape
        #
        # means = np.zeros((T,K,M,Dx))
        #
        # if self.weights.shape[1] == self.weights.shape[2]:
        #     eye_weights = self.weights + np.eye(self.weights.shape[1])[None, :, :, None]
        # else:
        #     eye_weights = self.weights + np.column_stack(
        #             (np.eye(self.weights.shape[1]), np.zeros((self.weights.shape[1], 1))))[None, :, :, None]
        #
        # for k in range(K):
        #     for m in range(M):
        #         means[:,k,m,:] = np.einsum('ij, tj -> ti', eye_weights[k,:,:,m], x)
        #
        # means = to_t(means)
        if self.weights.shape[1] == self.weights.shape[2]:
            means = to_t(np.einsum('kijm, tj -> tkmi', self.weights + np.eye(self.weights.shape[1])[None,:,:,None], x, optimize='optimal'))
        else:
            means = to_t(
                np.einsum('kijm, tj -> tkmi', self.weights + np.column_stack((np.eye(self.weights.shape[1]), np.zeros((self.weights.shape[1], 1))))[None,:,:,None], x, optimize='optimal'))
        covs = to_t(self.covs)

        log_likes = torch.distributions.MultivariateNormal(means, covs[None, :, None, :, :],
                                                           validate_args=False).log_prob(y[:, None, None, :])  # gives TxKxM log likelihoods
        T,K,M = log_likes.shape
        return log_likes.reshape((T,K*M)).numpy()

    def log_prior_likelihood(self):
            tau_grid_torch = to_t(self.taus)

            Kcov = self.weight_gp_kernel(tau_grid_torch) + kernel_ridge*torch.eye(self.num_taus)[None,:,:] # add small ridge for stability
            Kinv = torch.inverse(Kcov)

            A_tensor = to_t(self.weights) # num_z_states x data_dim x data_dim x num_tau_states
            # pdb.set_trace()

            # \sum_{ijk} -0.5 * a_ijk ' * inv(K_k) * a_ijk
            Kia = torch.matmul(Kinv, A_tensor.permute(0,3,1,2).flatten(2,3)) # now assuming A is K x D x D x M

            quad_term = -0.5 * torch.sum(Kia * A_tensor.permute(0,3,1,2).flatten(2,3))

            # \sum_k -0.5 * D^2 * log|K_k|

            log_det_term = -0.5 * self.data_dim**2 * torch.sum(Kcov.logdet()) 

            return quad_term + log_det_term


    def hyper_M_step(self, niter=100, learning_rate=1e-3):
        # function to optimize hyperparameters inside kernel object
        optimizer = torch.optim.SGD(self.weight_gp_kernel.parameters(), lr=learning_rate)
        # optimizer = torch.optim.LBFGS(self.weight_gp_kernel.parameters(), lr=learning_rate, history_size=100, line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            loss = -self.log_prior_likelihood()
            loss.backward()
            return loss
    
        for i in range(niter):
            optimizer.step(closure)



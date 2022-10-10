import numpy as np
import numpy.random as npr
import scipy.stats
import torch
from tqdm.auto import trange
from torch.distributions import MultivariateNormal
import pickle
import os
from util import random_rotation, sum_tuples
import wandb
import time
from numba import njit, prange

device = torch.device('cpu')
dtype = torch.float64
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

class TWARHMM(object):

    def __init__(self, config, taus=None):  #config is a dictionary containing parameters
        self.config = dict(config)
        self.num_discrete_states = config["num_discrete_states"]
        self.data_dim = config["data_dim"]
        self.covariates_dim = config["covariates_dim"]
        if np.any(taus == None): self.taus = np.logspace(-config["tau_scale"],config["tau_scale"],config["num_taus"],base=2)
        else: self.taus = taus
        if config["num_taus"] == 1:
            self.taus = np.array([1.])
        self.kappa = config["kappa"]
        self.alpha = config["alpha"]
        self.transitions = Transitions(self.num_discrete_states, len(self.taus), self.alpha, self.kappa, random_init=False)

        self.observations = LinearRegressionObservations(self.num_discrete_states, self.data_dim,
                                                    self.covariates_dim, self.taus, config["covariance_reg"])

    def fit(self, train_dataset, test_dataset, seed=0, num_epochs=50, fit_observations=True, fit_transitions=False, fit_tau_trans=False):
        # Fit using full batch EM
        num_train = sum([len(data["data"]) for data in train_dataset])
        num_test = sum([len(data["data"]) for data in test_dataset])
        # Initialize with a random posterior
        #posteriors = initialize_posteriors(train_dataset, self.num_discrete_states * self.taus.shape[0], seed=seed)
        total_states = self.num_discrete_states*len(self.taus)
        posteriors = [Posterior(self, data_dict, total_states) for data_dict in train_dataset]
        for posterior in posteriors:
            posterior.update()
        continuous_expectations, discrete_expectations = self.compute_expected_suff_stats(train_dataset, posteriors, self.taus, fit_observations, fit_transitions)
        train_lls = []
        test_lls = []

        # Main loop
        for itr in trange(num_epochs):
            print(itr)
            self.M_step(continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau_trans)

            for posterior in posteriors:
                posterior.update()

            # Compute the expected sufficient statistics under the new posteriors
            continuous_expectations, discrete_expectations = self.compute_expected_suff_stats(train_dataset, posteriors, self.taus, fit_observations, fit_transitions)

            # Store the average train likelihood
            avg_train_ll = sum([p.marginal_likelihood() for p in posteriors]) / num_train
            train_lls.append(avg_train_ll)

            # Compute the posteriors for the test dataset too
            test_posteriors = [Posterior(self,data_dict,total_states) for data_dict in test_dataset]

            for posterior in test_posteriors:
                posterior.update()

            # Store the average test likelihood
            avg_test_ll = sum([p.marginal_likelihood() for p in test_posteriors]) / num_test
            test_lls.append(avg_test_ll)

        # convert lls to arrays
        train_lls = np.array(train_lls)
        test_lls = np.array(test_lls)
        return train_lls, test_lls, posteriors, test_posteriors

    def fit_stoch(self, train_dataset, test_dataset, forgetting_rate=-0.5, seed=0, num_epochs=5, fit_observations=True,
                    fit_transitions=True, fit_tau_trans = True, compute_posteriors=True, wandb_log=True):
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
                self.M_step(continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau=fit_tau_trans)

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
                avg_mll = sum([p.marginal_likelihood() for p in posteriors]) / this_num_train
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
                avg_test_mll = sum([p.marginal_likelihood() for p in test_posteriors]) / num_test
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

    def save(self, filepath):
        # TODO: add optional artifact saving
        os.mkdir(filepath)
        obs_outfile = open(os.path.join(filepath, "model"), 'wb')
        pickle.dump(self, obs_outfile)
        obs_outfile.close()

    @staticmethod
    def load(dir):
        model_infile = open(os.path.join(dir, "model"), 'rb')
        model = pickle.load(model_infile)
        model_infile.close()
        return model

    @staticmethod
    def load_wnb(artifact_filepath):
        artifact = wandb.use_artifact(artifact_filepath, type="model")
        artifact_dir = artifact.download()
        return TWARHMM.load(artifact_dir)


    def E_step(self,initial_dist, transition_matrix, log_likes, compute_joints=True):
        (Pz,Pt) = transition_matrix

        max_factor = np.max(log_likes, axis=1, keepdims=True)
        alphas, marginal_ll = self.nb_forward_pass(initial_dist, transition_matrix, log_likes,max_factor)

        betas = self.nb_backward_pass(transition_matrix, log_likes, max_factor)

        likes_tilde = np.exp(log_likes - np.max(log_likes, axis=1)[:, None])
        hadamard_prod = alphas * likes_tilde * betas
        expected_states = hadamard_prod / np.sum(hadamard_prod, axis=1)[:, None]

        alphas = alphas.reshape((alphas.shape[0],self.num_discrete_states,len(self.taus)))
        betas = betas.reshape((betas.shape[0], self.num_discrete_states, len(self.taus)))
        log_likes = log_likes.reshape((log_likes.shape[0],self.num_discrete_states, len(self.taus)))

        if compute_joints: #TODO: split into 2 matrices
            alphas_z = alphas.sum(axis=2)
            alphas_t = alphas.sum(axis=1)
            betas_z = betas.sum(axis=2)
            betas_t = betas.sum(axis=1)
            log_likes_z = log_likes.sum(axis=2)
            log_likes_t = log_likes.sum(axis=1)
            likes_tilde_z = np.exp(log_likes_z - np.max(log_likes_z, axis=1)[:, None])
            likes_tilde_t = np.exp(log_likes_t - np.max(log_likes_t, axis=1)[:, None])

            hadamard_2_z = alphas_z[:-1, :, None] * likes_tilde_z[:-1, :, None] * likes_tilde_z[1:, None,:] * Pz[None, :, :] * betas_z[1:,None,:]
            expected_joints_z = hadamard_2_z / np.sum(hadamard_2_z, axis=(1, 2), keepdims=True)

            hadamard_2_t = alphas_t[:-1, :, None] * likes_tilde_t[:-1, :, None] * likes_tilde_t[1:, None, :] * Pt[None,:,:] * betas_t[1:,None,:]
            expected_joints_t = hadamard_2_t / np.sum(hadamard_2_t, axis=(1, 2), keepdims=True)

            expected_joints = (expected_joints_z,expected_joints_t)
        else:
            expected_joints = (None, None)

        # Package the results into a dictionary summarizing the posterior
        posterior = dict(expected_states=expected_states,
                         expected_joints=expected_joints,
                         marginal_ll=marginal_ll)
        return posterior

    def M_step(self, continuous_expectations, discrete_expectations, fit_observations, fit_transitions, fit_tau):
        if fit_transitions: self.transitions.M_step(discrete_expectations, fit_tau=fit_tau)
        if fit_observations: self.observations.M_step(continuous_expectations)

    def forward_pass(self, initial_dist, transition_matrix, log_likes):
        (Pz,Pt) = transition_matrix
        alphas = np.zeros_like(log_likes)
        marginal_ll = 0
        T = log_likes.shape[0]
        max_factor = np.max(log_likes, axis=1, keepdims=True)
        likes_tilde = np.exp(log_likes - max_factor)

        alphas[0] = np.squeeze(initial_dist)

        for t in range(1, T):
            A_t_minus_1 = np.sum(alphas[t - 1] * likes_tilde[t - 1], axis=-1)
            # alphas[t] = (1 / A_t_minus_1) * \
            #             transition_matrix.T @ (alphas[t - 1] * likes_tilde[t - 1])
            alphas[t] = (1 / A_t_minus_1) * \
                        np.einsum('ab,bc,cd->ad',Pz.T,np.reshape(alphas[t - 1] * likes_tilde[t - 1],(Pz.shape[0],Pt.shape[0])),Pt).ravel()
            if A_t_minus_1 > 0 and not np.any(np.isnan(A_t_minus_1)):
                marginal_ll += np.sum(np.log(A_t_minus_1) + max_factor[t - 1])
            else:
                print("yikes")

        A_t = np.sum(alphas[t] * likes_tilde[t], axis=-1)
        marginal_ll += np.sum(np.log(A_t) + max_factor[t])

        return alphas, marginal_ll

    def backward_pass(self, transition_matrix, log_likes):
        (Pz,Pt) = transition_matrix
        betas = np.zeros_like(log_likes)
        T, K = log_likes.shape
        max_factor = np.max(log_likes, axis=1, keepdims=True)
        likes_tilde = np.exp(log_likes - max_factor)

        betas[T - 1] = 1 / K

        for t in range(T - 2, -1, -1):  # iterate from T-2 ==> 0
            #betas[t] = transition_matrix @ (betas[t + 1] * likes_tilde[t + 1])
            betas[t] = np.einsum('ab,bc,cd->ad',Pz,np.reshape(betas[t + 1] * likes_tilde[t + 1],(Pz.shape[0],Pt.shape[0])),Pt.T).ravel()
            betas[t] /= np.sum(betas[t])  # normalize before the next step

        return betas

    def compute_expected_suff_stats(self, dataset, posteriors, taus, fit_observations, fit_transitions):
        assert isinstance(dataset, list)
        assert isinstance(posteriors, list)

        # Helper function to compute expected counts and sufficient statistics
        # for a single time series and corresponding posterior.
        def _compute_expected_suff_stats(data, posterior, taus, fit_observations, fit_transitions):
            Dx = data["data"].shape[1]
            D = data["covariates"].shape[1]
            q = posterior.expected_states()
            (fancy_e_z, fancy_e_t) = posterior.expected_transitions() #TODO: change to return two matrices
            q += 1e-16
            q = q / q.sum(axis=1, keepdims=True)  # basically Laplace smoothing
            L = taus.shape[0]
            K = q.shape[1] / L
            q = q.reshape((q.shape[0], int(K), L))  # dim TxKxL
            K = q.shape[1]
            dxxT_Etau = np.zeros((K, Dx, D))
            xxT = np.zeros((K, D, D))
            dxdxT_Etau2 = np.zeros((K, Dx, Dx))
            T = np.zeros(K)
            fancy_e_z_over_T = np.zeros((self.num_discrete_states, self.num_discrete_states))
            fancy_e_t_over_T = np.zeros((len(self.taus), len(self.taus)))
            q_one = np.zeros(self.num_discrete_states * len(self.taus))
            for k in range(K):
                qzt = q[:, k, :].sum(axis=-1)

                if fit_observations:
                    #TODO: rewrite with descriptive variable names
                    q_taugivenz = q[:, k, :] / np.sum(q[:, k, :], axis=-1, keepdims=True)
                    E_tau_given_k = np.einsum('tl,l -> t', q_taugivenz, taus)  # TxL and L -> T
                    E_tauinv_given_k = np.einsum('tl,l -> t', q_taugivenz, (1/taus))  # TxL and L -> T

                    # sufficient stats for A
                    dxxT_Etau[k, :, :] = np.einsum('t,tij->ij', qzt, data['suff_stats'][2])
                    xxT[k, :, :] = np.einsum('t,t,tij->ij', qzt, E_tauinv_given_k, data['suff_stats'][3])

                    # sufficient stats for Q
                    dxdxT_Etau2[k, :, :] = np.einsum('t,t,tij->ij', qzt, E_tau_given_k, data['suff_stats'][1])


                T[k] = np.dot(qzt, data['suff_stats'][0])

            if fit_transitions:
                fancy_e_z_over_T = np.einsum('tij->ij', fancy_e_z)
                fancy_e_t_over_T = np.einsum('tij->ij', fancy_e_t)

                q_one = posterior.expected_states()[0]

            stats = (tuple((dxxT_Etau, xxT, dxdxT_Etau2, T)),
                     tuple((fancy_e_z_over_T, fancy_e_t_over_T, q_one)))
            return stats

        # Sum the expected stats over the whole dataset
        stats = (None,None)
        for data, posterior in zip(dataset, posteriors):
            these_stats = _compute_expected_suff_stats(data, posterior, taus, fit_observations, fit_transitions)
            stats_cont = sum_tuples(stats[0], these_stats[0])
            stats_disc = sum_tuples(stats[1], these_stats[1])
            stats = (stats_cont, stats_disc)
        return stats

    def sample(self, T, bias=False): #TODO: might only work for relatively low total states
        observations = self.observations
        initial_dist = self.transitions.initial_dist
        (Pz,Pt) = self.transitions.transition_matrix
        transition_matrix = np.kron(Pz,Pt)
        taus = self.taus
        if bias:
            x = np.hstack((np.zeros((T, observations.data_dim)),np.ones((T,1))))
        else:
            x = np.zeros((T, observations.data_dim))
        z = np.zeros((T), dtype=np.int)
        num_states = initial_dist.shape[0]
        z[0] = np.random.choice(range(initial_dist.shape[0]), p=initial_dist)
        
        timescaled_weights, timescaled_covs = self.observations.timescale_weights_covs(observations.weights, observations.covs, taus)
        if bias:
            x[0,:-1] = MultivariateNormal(to_t(np.zeros(observations.data_dim)), to_t(timescaled_covs[z[0], :, :])).sample()
        else:
            x[0] = MultivariateNormal(to_t(np.zeros(observations.data_dim)), to_t(timescaled_covs[z[0], :, :])).sample()
        for i in range(1, T):
            z[i] = np.random.choice(range(num_states), p=transition_matrix[z[i - 1], :])
            # mu = timescaled_weights[z[i], :, :-1]@x[i-1] + timescaled_weights[z[i], :, -1] #changed to account for no bias
            mu = timescaled_weights[z[i], :, :] @ x[i - 1]
            cov = timescaled_covs[z[i], :, :]
            if bias:
                x[i,:-1] = MultivariateNormal(to_t(mu), to_t(cov)).sample()
            else:
                x[i] = MultivariateNormal(to_t(mu), to_t(cov)).sample()
        if bias:
            x = x[:,:-1]
        return z, x

    @staticmethod
    @njit()
    def nb_forward_pass(initial_dist, transition_matrix, log_likes, max_factor):
        (Pz,Pt) = transition_matrix
        alphas = np.zeros_like(log_likes)
        marginal_ll = 0
        T = log_likes.shape[0]
        likes_tilde = np.exp(log_likes - max_factor)

        alphas[0] = initial_dist

        for t in range(1, T):
            A_t_minus_1 = np.sum(alphas[t - 1] * likes_tilde[t - 1])
            alphas[t] = (1 / A_t_minus_1) * \
                        (Pz.T @ (np.reshape(alphas[t - 1] * likes_tilde[t - 1],(Pz.shape[0],Pt.shape[0]))) @ Pt).ravel()
            # alphas[t] = (1 / A_t_minus_1) * \
            #             np.einsum('ab,bc,cd->ad',Pz.T,np.reshape(alphas[t - 1] * likes_tilde[t - 1],(Pz.shape[0],Pt.shape[0])),Pt).ravel()
            # if A_t_minus_1 > 0 and not np.any(np.isnan(A_t_minus_1)):
            marginal_ll += np.sum(np.log(A_t_minus_1) + max_factor[t - 1])
            # else:
            #     print("yikes")

        A_t = np.sum(alphas[t] * likes_tilde[t])
        marginal_ll += np.sum(np.log(A_t) + max_factor[t])

        return alphas, marginal_ll

    @staticmethod
    @njit()
    def nb_backward_pass(transition_matrix, log_likes, max_factor):
        (Pz,Pt) = transition_matrix
        betas = np.zeros_like(log_likes)
        T, K = log_likes.shape
        likes_tilde = np.exp(log_likes - max_factor)

        betas[T - 1] = 1 / K

        for t in range(T - 2, -1, -1):  # iterate from T-2 ==> 0
            betas[t] = (Pz @ (np.reshape(betas[t + 1] * likes_tilde[t + 1],(Pz.shape[0],Pt.shape[0]))) @ Pt.T).ravel()
            #betas[t] = np.einsum('ab,bc,cd->ad',Pz,np.reshape(betas[t + 1] * likes_tilde[t + 1],(Pz.shape[0],Pt.shape[0])),Pt.T).ravel()
            betas[t] /= np.sum(betas[t])  # normalize before the next step

        return betas

class LinearRegressionObservations(object):
    """
    Wrapper for a collection of Gaussian observation parameters.
    """

    def __init__(self, num_states, data_dim, covariate_dim, taus, covariance_reg, random_weights=True):
        """
        Initialize a collection of observation parameters for an HMM whose
        observation distributions are linear regressions. The HMM has
        `num_states` (i.e. K) discrete states, `data_dim` (i.e. D)
        dimensional observations, and `covariate_dim` covariates.
        In an ARHMM, the covariates will be functions of the past data.

        Note: self.weights is always the continuous time operator.
        """
        self.num_states = num_states
        self.data_dim = data_dim
        self.covariate_dim = covariate_dim
        self.taus = taus
        self.covariance_reg = covariance_reg

        # Initialize the model parameters
        if random_weights:
            self.weights = np.zeros((num_states, data_dim, covariate_dim))
            for i in range(num_states):
                self.weights[i,:,:data_dim] = scipy.linalg.logm(random_rotation(data_dim,theta= np.pi/20))
        else:
            self.weights = np.zeros((num_states, data_dim, covariate_dim))
        #TODO: do we need this scaling?
        self.covs = .05*np.tile(np.eye(data_dim), (num_states, 1, 1))

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
        Nothing, but the dataset is updated in place to have a new `suff_stats`
            key, which contains a tuple of sufficient statistics.
        """
        ###
        # YOUR CODE BELOW
        #
        for data in dataset:
            x = data['data']
            phi = data['covariates']
            #TODO: update to generalize for lags >1
            if x.shape[1] == phi.shape[1]: #no bias
                dx = x - phi
            else:
                dx = x - phi[:,:-1]
            data['suff_stats'] = (np.ones(len(x)),
                                  np.einsum('ti,tj->tij', dx, dx), # dxn dxn.T
                                  np.einsum('ti,tj->tij', dx, phi), # dxn xn-1.T
                                  np.einsum('ti,tj->tij', phi, phi)) # xn-1 xn-1.T
        #
        ###

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
        taus = self.taus

        timescaled_weights, timescaled_covs = self.timescale_weights_covs(self.weights,self.covs,taus)
        means = to_t(timescaled_weights @ x.T)
        covs = to_t(timescaled_covs)

        K, _, _ = means.shape
        T, _ = x.shape
        log_likes = np.zeros((T, K))
        for k in range(K):
            dist = torch.distributions.MultivariateNormal(means[k].T, covs[k],validate_args=False)
            log_likes[:, k] = dist.log_prob(y)
        #
        return log_likes

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
        dxxT_Etau, xxT, dxdxT_Etau2, T = continuous_expectations
        ###
        for k in range(self.num_states):
            AstarT = np.linalg.solve(xxT[k], dxxT_Etau[k].T)
            self.weights[k] = AstarT.T  #continuous time operator (unscaled)
            self.covs[k] = self.covariance_reg* np.eye(self.data_dim) + \
                           (dxdxT_Etau2[k] - dxxT_Etau[k] @ AstarT - AstarT.T @ dxxT_Etau[k].T + AstarT.T @ xxT[k] @ AstarT) / T[k]

    @classmethod
    def timescale_weights_covs(cls, weights,covs,taus):
        '''
        scale continuous time operator
        '''
        tiled_weights = np.repeat(weights,len(taus),axis=0)
        tiled_taus = np.tile(taus,weights.shape[0])
        if weights.shape[1] == weights.shape[2]:
            timescaled_weights = np.eye(weights.shape[1]) + tiled_weights/tiled_taus[:,None,None]
        else:
            timescaled_weights = np.hstack((np.eye(weights.shape[1]),np.zeros((weights.shape[1],1)))) + tiled_weights / tiled_taus[:, None, None]
        tiled_covs = np.repeat(covs, len(taus), axis=0)
        timescaled_covs = tiled_covs/tiled_taus[:,None,None]
        return timescaled_weights, timescaled_covs

class Transitions(object):
    def __init__(self, num_discrete_states, num_taus, alpha, kappa, random_init=True):
        self.num_discrete_states = num_discrete_states
        self.num_taus = num_taus
        self.initial_dist = np.ones(self.num_discrete_states*self.num_taus) / (self.num_discrete_states*self.num_taus)
        if random_init:
            Pz = .99 * np.eye(self.num_discrete_states) + .01 * npr.rand(self.num_discrete_states,
                                                                         self.num_discrete_states)
            Pz /= Pz.sum(axis=1, keepdims=True)
            Pt = .95 * np.eye(self.num_taus) + .05 * npr.rand(self.num_taus, self.num_taus)
            Pt /= Pt.sum(axis=1, keepdims=True)
        else:
            if self.num_discrete_states != 1:
                Pz = .99 * np.eye(self.num_discrete_states) + .01/(self.num_discrete_states-1) * (np.ones((self.num_discrete_states,
                                                                             self.num_discrete_states))-np.eye(self.num_discrete_states))
            else: Pz = np.array([[1.]])
            if self.num_taus != 1:
                Pt = .95 * np.eye(self.num_taus) + .025 * (np.diag(np.ones(self.num_taus-1), 1) + np.diag(np.ones(self.num_taus-1), -1))
                Pt /= Pt.sum(axis=1, keepdims=True)
            else: Pt = np.array([[1.]])
        self.transition_matrix = (Pz,Pt)
        self.alpha = alpha
        self.kappa = kappa

    def M_step(self, discrete_expectations, fit_z = True, fit_tau = True): #TODO: kron first pass is done
        expected_joints_z, expected_joints_t, q_zero = discrete_expectations
        if fit_z:
            expected_joints_z += self.kappa * np.eye(self.num_discrete_states) + (self.alpha-1) * np.ones((self.num_discrete_states, self.num_discrete_states))
            expected_joints_z += 1e-16
            Pz = np.nan_to_num(expected_joints_z / expected_joints_z.sum(axis=1, keepdims=True))
        else: Pz = self.transition_matrix[0]

        if fit_tau:
            expected_joints_t += self.kappa * np.eye(self.num_taus) + (self.alpha - 1) * np.ones((self.num_taus, self.num_taus))
            expected_joints_t += 1e-16
            Pt = np.nan_to_num(expected_joints_t / expected_joints_t.sum(axis=1, keepdims=True))
        else:
            Pt = self.transition_matrix[1]
        self.transition_matrix = (Pz,Pt)
        self.initial_dist = q_zero / np.sum(q_zero, keepdims=True)

class Posterior(object):

    def __init__(self, model, data, num_states, seed=0):
        self.model = model
        self.data = data
        self.num_states = num_states
        self.num_taus = len(self.model.taus)
        self.num_discrete_states = self.model.num_discrete_states
        self._posterior = self._initialize_posteriors(data, num_states, seed)

    def _initialize_posteriors(self, dataset, num_states, seed=0):
        # rng = npr.RandomState(seed)
        # expected_states = rng.rand(len(dataset["data"]), num_states)
        # expected_states /= expected_states.sum(axis=1, keepdims=True)
        expected_taus = np.ones(
            (len(dataset["data"]), num_states, 2))  # mu, sigma for each time step and each discrete state
        # expected_joints = rng.rand(len(dataset["data"]) - 1, num_states, num_states)
        # expected_joints /= expected_joints.sum(axis=(1, 2), keepdims=True)
        expected_states = np.zeros((len(dataset["data"]), num_states))
        # expected_joints = (np.zeros((len(dataset["data"]) - 1, self.model.num_discrete_states, self.model.num_discrete_states)),
        #                    np.zeros((len(dataset["data"]) - 1, len(self.model.taus),
        #                              len(self.model.taus))))
        expected_joints = (np.zeros((len(dataset["data"]), self.num_discrete_states,self.num_discrete_states)),
                           np.zeros((len(dataset["data"]),self.num_taus,self.num_taus)))
        return dict(expected_states=expected_states,
                               expected_joints=expected_joints,
                               marginal_ll=-np.inf)

    def update(self):
        """
        Run the exact message passing algorithm to infer the posterior distribution.
        """

        log_likes = self.model.observations.log_likelihoods(self.data)
        #should throw error if compute_joints is False while trying to update transitions
        #TODO: better way to handle compute_joints argument
        new_posterior = self.model.E_step(self.model.transitions.initial_dist, self.model.transitions.transition_matrix, log_likes, compute_joints=True)
        self._posterior = new_posterior
        return self

    def get_states(self):
        # assumes posterior is already updated
        # TODO: replace with Viterbi
        # currently: for every z_t, find max q(z_t| x_1:T)
        # goal: max z_1:T q(z_1:T| x_1:T)
        return self._posterior['expected_states'].argmax(1)

    def marginal_likelihood(self):
        """Compute the marginal likelihood of the data under the model.
        Returns:
            ``\log p(x_{1:T})`` the marginal likelihood of the data
            summing over discrete latent state sequences.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["marginal_ll"]

    def expected_states(self):
        """Compute the expected values of the latent states under the
        posterior distribution.
        Returns:
            ``E[z_t | x_{1:T}]`` the expected value of the latent state
                at time ``t`` given the sequence of data.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["expected_states"]

    def expected_transitions(self):
        """Compute the expected transitions of the latent states under the
        posterior distribution.
        Returns:
            ``E[z_t z_{t+1} | x_{1:T}]`` the expected value of
                adjacent latent states given the sequence of data.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["expected_joints"]

    @staticmethod
    def state_durations(states, total_states):
        changepoints = states != np.hstack((states[1:], -1))  # 1 where state change occurs
        changepoint_frame = np.where(changepoints)[0]  # timestamps of changepoints
        changepoint_states = states[changepoints]  # state label of changepoint
        state_durations = np.diff(np.hstack((0, changepoint_frame)))  # duration before each change
        state_durations[0] += 1
        durations = []
        for k in range(total_states):
            changepoint_indices = changepoint_states == k
            durations.append(state_durations[changepoint_indices])
        return durations

    def state_usage(self):
        states = self.get_states()
        return np.bincount(states, minlength=self.num_states)

    def state_switch(self):
        states = self.get_states()
        changepoints = states != np.hstack((states[1:], -1))  # 1 where state change occurs
        changepoint_states = states[changepoints]  # state label of changepoint
        return changepoint_states

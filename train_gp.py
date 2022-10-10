import wandb
import numpy as np
import os
from warhmm_gp import TWARHMM_GP, LinearRegressionObservations_GP
from data_util import load_dataset, standardize_pcs, precompute_ar_covariates, log_wandb_model
import datetime
from kernels import RBF
import matplotlib.pyplot as plt

data_dim = 10
num_lags = 1

hyperparameter_defaults = dict(
    num_discrete_states=20,
    data_dim=data_dim,
    covariates_dim=11,
    tau_scale=1,
    num_taus=31,
    kappa=10000,
    alpha=5,
    covariance_reg=1e-4,
    lengthscale=1
)

train_dataset, test_dataset = load_dataset(num_pcs=data_dim)
train_dataset, mean, std = standardize_pcs(train_dataset)
test_dataset, _, _ = standardize_pcs(test_dataset, mean, std)

print("data loaded")
# First compute the autoregression covariates
precompute_ar_covariates(train_dataset, num_lags=num_lags, fit_intercept=True)
precompute_ar_covariates(test_dataset, num_lags=num_lags, fit_intercept=True)

# Then precompute the sufficient statistics
LinearRegressionObservations_GP.precompute_suff_stats(train_dataset)
LinearRegressionObservations_GP.precompute_suff_stats(test_dataset)

covariates_dim = train_dataset[0]['covariates'].shape[1]

projectname = "twarhmm_gp"
wandb.init(config=hyperparameter_defaults, entity="twss", project=projectname)
config = wandb.config

taus = np.linspace(-config['tau_scale'], config['tau_scale'], config['num_taus'])
twarhmm_gp = TWARHMM_GP(config, taus, kernel=RBF(config['num_discrete_states'], config['lengthscale']))

train_lls, test_lls, train_posteriors, test_posteriors, = \
        twarhmm_gp.fit_stoch(train_dataset,
                         test_dataset,
                          num_epochs=50, fit_transitions=True, fit_tau=False, fit_kernel_params=False, wandb_log=True)
#plt.plot(test_lls)
# e = datetime.datetime.now()
#
log_wandb_model(twarhmm_gp, "twarhmm_gp_K{}_T{}".format(twarhmm_gp.num_discrete_states,len(twarhmm_gp.taus)),type="model")
# if test_posteriors is not None:
#     wnb_histogram_plot(test_posteriors, tau_duration=True, duration_plot=True, state_usage_plot=True, ordered_state_usage=True, state_switch=True)
#     centroid_velocity_plot(test_posteriors)
# #save_videos_wandb(test_posteriors)
#
wandb.finish()
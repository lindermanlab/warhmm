import wandb
import os
from twarhmm import TWARHMM, LinearRegressionObservations
from plotting_util import wnb_histogram_plot, save_videos_wandb, centroid_velocity_plot
from data_util import load_dataset, standardize_pcs, precompute_ar_covariates, log_wandb_model
import datetime

data_dim = 10
num_lags = 1

hyperparameter_defaults = dict(
    num_discrete_states=5,
    data_dim=data_dim,
    covariates_dim=11,
    tau_scale=0.6,
    num_taus=5,
    kappa=10000,
    alpha=5,
    covariance_reg=1e-4)

train_dataset, test_dataset = load_dataset(num_pcs=data_dim)
train_dataset, mean, std = standardize_pcs(train_dataset)
test_dataset, _, _ = standardize_pcs(test_dataset, mean, std)

print("data loaded")
# First compute the autoregression covariates
precompute_ar_covariates(train_dataset, num_lags=num_lags, fit_intercept=True)
precompute_ar_covariates(test_dataset, num_lags=num_lags, fit_intercept=True)

# Then precompute the sufficient statistics
LinearRegressionObservations.precompute_suff_stats(train_dataset)
LinearRegressionObservations.precompute_suff_stats(test_dataset)

covariates_dim = train_dataset[0]['covariates'].shape[1]

projectname = "twarhmm"
wandb.init(config=hyperparameter_defaults, entity="twss", project=projectname)
config = wandb.config

twarhmm = TWARHMM(config, None)

train_lls, test_lls, train_posteriors, test_posteriors, = \
        twarhmm.fit_stoch(train_dataset,
                         test_dataset,
                          num_epochs=50, compute_posteriors=True, fit_transitions=True)

e = datetime.datetime.now()

log_wandb_model(twarhmm, "twarhmm_K{}_T{}".format(twarhmm.num_discrete_states,len(twarhmm.taus)),type="model")
if test_posteriors is not None:
    wnb_histogram_plot(test_posteriors, tau_duration=True, duration_plot=True, state_usage_plot=True, ordered_state_usage=True, state_switch=True)
    centroid_velocity_plot(test_posteriors)
#save_videos_wandb(test_posteriors)

wandb.finish()

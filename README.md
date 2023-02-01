# WARHMM: distinguishing discrete and continuous behavioral variability

This repository contains an implementation of different classes of Warped Autoregressive Hidden Markov Models (WARHMM). The associated paper is available [here](https://openreview.net/forum?id=6Kj1wCgiUp_).

## File descriptions
```
data_util.py             Dataset loader + preprocessing functions
kernels.py               Kernel functions for GP-WARHMM
plotting_util.py         Assortment of possibly helpful plotting functions
train.py                 A script for training time-warped ARHMM
train_gp.py              A script for training GP-warped ARHMM
twarhmm.py               Time-warped ARHMM model class
util.py                  Utility functions
warhmm_gp.py             GP-warped ARHMM model class
 ```
 ## How to run?
 Load data, initialize a model (either GP-WARHMM or T-WARHMM), and train:
 ```
 # assume train_dataset and test_dataset are loaded (see note below)
 # standardize PCs
train_dataset, mean, std = standardize_pcs(train_dataset)
test_dataset, _, _ = standardize_pcs(test_dataset, mean, std)
 
# first compute the autoregression covariates
precompute_ar_covariates(train_dataset, num_lags=num_lags, fit_intercept=True)
precompute_ar_covariates(test_dataset, num_lags=num_lags, fit_intercept=True)

# then precompute the sufficient statistics
LinearRegressionObservations.precompute_suff_stats(train_dataset)
LinearRegressionObservations.precompute_suff_stats(test_dataset)

# set model hyperparameters
config = dict(
    num_discrete_states=20,
    data_dim=data_dim,
    covariates_dim=11,
    tau_scale=0.6,
    num_taus=31,
    kappa=1e10,
    alpha=1,
    covariance_reg=1e-4,
)

# initialize model 
twarhmm = TWARHMM(config, None)

# fit the thing!
train_lls, test_lls, train_posteriors, test_posteriors, = \
        twarhmm.fit_stoch(train_dataset,
                         test_dataset,
                          num_epochs=10, fit_transitions=True, fit_tau_trans=False, wandb_log=True)
 ```
 
 ## A note on the data
 The MoSeq dataset is available in combination with the original MoSeq code at [the Datta Lab's website](https://dattalab.github.io/moseq2-website/). Synthetic data can be generated from the T-WARHMM using the sample() function in twarhmm.py.
 
 If you would like to use WARHMM on your own data, we assume loaded data is formatted as follows:
 - train_data (or test_data) is a list of dictionaries of data
 - an entry of train_data is a dictionary of data associated with a specific animal/trial, and contains at least the key 'raw_pcs' which is associated with a numpy array of shape T(ime) x D(imensions)
 - if your data is contained in a single dictionary just wrap it in a list and it should run -- please reach out with any questions
 
 ## Citation
 ```
@inproceedings{costacurta2022warhmm,
  title={Distinguishing discrete and continuous behavioral variability using warped autoregressive HMMs},
  author={Julia Costacurta and Lea Duncker and Blue Sheffer and Winthrop Gillis and Caleb Weinreb and Jeffrey Markowitz and Sandeep Datta and Alex Williams and Scott Linderman},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=6Kj1wCgiUp_}
}
```

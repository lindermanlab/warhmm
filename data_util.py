from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
from tqdm.auto import trange
from pynwb import NWBHDF5IO
import seaborn as sns
import wandb
import os

def load_dataset(indices=None,load_frames=True,num_pcs=10,datapath='moseq_data/saline_example',datapath2='amph_data/amphetamine_example',mix_data=False):
    '''
        loads mouse data and returns train/test datasets as dicts
    '''

    if indices is None:
        indices = np.arange(24)

    if ~mix_data:
        train_dataset = []
        test_dataset = []
        for t in trange(len(indices)):
            i = indices[t]
            nwb_path = datapath + "_{}.nwb".format(i)
            with NWBHDF5IO(nwb_path, mode='r') as io:
                f = io.read()
                num_frames = len(f.processing['MoSeq']['PCs']['pcs_clean'].data)
                train_slc = slice(0, int(0.8 * num_frames))
                test_slc = slice(int(0.8 * num_frames) + 1, -1)

                train_data, test_data = dict(), dict()
                for slc, data in zip([train_slc, test_slc], [train_data, test_data]):
                    data["raw_pcs"] = f.processing['MoSeq']['PCs']['pcs_clean'].data[slc][:, :num_pcs]
                    data["times"] = f.processing['MoSeq']['PCs']['pcs_clean'].timestamps[slc][:]
                    data["centroid_x_px"] = f.processing['MoSeq']['Scalars']['centroid_x_px'].data[slc][:]
                    data["centroid_y_px"] = f.processing['MoSeq']['Scalars']['centroid_y_px'].data[slc][:]
                    data["angles"] = f.processing['MoSeq']['Scalars']['angle'].data[slc][:]
                    data["labels"] = f.processing['MoSeq']['Labels']['labels_clean'].data[slc][:]
                    data["velocity_3d_px"] = f.processing['MoSeq']['Scalars']['velocity_3d_px'].data[slc][:]
                    data["height_ave_mm"] = f.processing['MoSeq']['Scalars']['height_ave_mm'].data[slc][:]

                # only load the frames on the test data
                test_data["frames"] = f.processing['MoSeq']['Images']['frames'].data[test_slc]

            train_dataset.append(train_data)
            test_dataset.append(test_data)

    elif mix_data:
        train_dataset = []
        test_dataset = []
        # ind_1 = np.random.randint(0,len(indices),len(indices)//2)
        # ind_2 = np.random.randint(0,len(indices),len(indices)//2)

        for t in trange(len(indices)):
            i = indices[t]
            nwb_path = datapath + "_{}.nwb".format(i)
            with NWBHDF5IO(nwb_path, mode='r') as io:
                f = io.read()
                num_frames = len(f.processing['MoSeq']['PCs']['pcs_clean'].data)
                train_slc = slice(0, int(0.8 * num_frames))
                test_slc = slice(int(0.8 * num_frames) + 1, -1)

                train_data, test_data = dict(), dict()
                for slc, data in zip([train_slc, test_slc], [train_data, test_data]):
                    data["raw_pcs"] = f.processing['MoSeq']['PCs']['pcs_clean'].data[slc][:, :num_pcs]
                    data["times"] = f.processing['MoSeq']['PCs']['pcs_clean'].timestamps[slc][:]
                    data["centroid_x_px"] = f.processing['MoSeq']['Scalars']['centroid_x_px'].data[slc][:]
                    data["centroid_y_px"] = f.processing['MoSeq']['Scalars']['centroid_y_px'].data[slc][:]
                    data["angles"] = f.processing['MoSeq']['Scalars']['angle'].data[slc][:]
                    data["labels"] = f.processing['MoSeq']['Labels']['labels_clean'].data[slc][:]

                # only load the frames on the test data
                test_data["frames"] = f.processing['MoSeq']['Images']['frames'].data[test_slc]

            train_dataset.append(train_data)
            test_dataset.append(test_data)

        for t in trange(len(indices)):
            i = indices[t]
            nwb_path = datapath2 + "_{}.nwb".format(i)
            with NWBHDF5IO(nwb_path, mode='r') as io:
                f = io.read()
                num_frames = len(f.processing['MoSeq']['PCs']['pcs_clean'].data)
                train_slc = slice(0, int(0.8 * num_frames))
                test_slc = slice(int(0.8 * num_frames) + 1, -1)

                train_data, test_data = dict(), dict()
                for slc, data in zip([train_slc, test_slc], [train_data, test_data]):
                    data["raw_pcs"] = f.processing['MoSeq']['PCs']['pcs_clean'].data[slc][:, :num_pcs]
                    data["times"] = f.processing['MoSeq']['PCs']['pcs_clean'].timestamps[slc][:]
                    data["centroid_x_px"] = f.processing['MoSeq']['Scalars']['centroid_x_px'].data[slc][:]
                    data["centroid_y_px"] = f.processing['MoSeq']['Scalars']['centroid_y_px'].data[slc][:]
                    data["angles"] = f.processing['MoSeq']['Scalars']['angle'].data[slc][:]
                    data["labels"] = f.processing['MoSeq']['Labels']['labels_clean'].data[slc][:]

                # only load the frames on the test data
                test_data["frames"] = f.processing['MoSeq']['Images']['frames'].data[test_slc]

            train_dataset.append(train_data)
            test_dataset.append(test_data)

    return train_dataset, test_dataset

def standardize_pcs(dataset, mean=None, std=None):
    '''
    adds new keyword 'data' corresponding with standardized PCs
    '''

    if mean is None and std is None:
        all_pcs = np.vstack([data['raw_pcs'] for data in dataset])
        mean = all_pcs.mean(axis=0)
        std = all_pcs.std(axis=0)

    for data in dataset:
        data['data'] = (data['raw_pcs'] - mean) / std
    return dataset, mean, std

def precompute_ar_covariates(dataset,
                             num_lags=1,
                             fit_intercept=False):
    '''
    add the desired covariates to the data dictionary
    '''
    for data in dataset:
        x = data['data']
        data_dim = x.shape[1]
        phis = []
        for lag in range(1, num_lags+1):
            phis.append(np.row_stack([np.zeros((lag, data_dim)), x[:-lag]]))
        if fit_intercept:
            phis.append(np.ones(len(x)))
        data['covariates'] = np.column_stack(phis)

def log_wandb_model(model, name, type):
    trained_model_artifact = wandb.Artifact(name,type=type)
    if not os.path.isdir('models'): os.mkdir('models')
    subdirectory = wandb.run.name
    filepath = os.path.join('models', subdirectory)
    model.save(filepath)
    trained_model_artifact.add_dir(filepath)
    wandb.log_artifact(trained_model_artifact)
# -*- coding:utf-8 -*-

import numpy as np
import mxnet as mx

import pandas as pd
import h5py
from scipy.interpolate import interp1d
from .utils import get_sample_indices


from lib.utils import pems_bay_mask_generator, Electricity_mask_generator, infer_mask, compute_mean


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    merge: boolean, default False,
           whether to merge training set and validation set to train model
    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    if graph_signal_matrix_filename == 'Electricity':
        path = 'data\Electricity_seqlen1_00masked\datasets.h5'
        with h5py.File(path, "r") as hf:
            # read data from h5 file
            X_train = hf['train']["X"][:]
            X_train = X_train[:, 0, :]
            x_train_mask = (np.isnan(X_train)).astype('uint8')
            X_train = np.where(x_train_mask == 1, 0, X_train)
        with h5py.File(path, "r") as hf:  # read data from h5 file
            X_val = hf['val']["X"][:, 0, :]
        with h5py.File(path, "r") as hf:  # read data from h5 file
            X_test = hf['test']["X"][:, 0, :]

        data_seq1 = np.concatenate((X_train, X_val, X_test), axis=0)
        mask = Electricity_mask_generator(data_seq1.shape, p_fault=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4)

        total_mask = mask
        data_seq = np.expand_dims(data_seq1, 2)
        mask = np.expand_dims(mask, 2)

        # fitting first time
        y_hat = []

        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0

        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)

        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask = np.transpose(np.array(y_hat))

        data_seq_mask = np.expand_dims(data_seq_mask, 2)

        all_samples = []


        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, data_seq_mask, mask, mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))


    if graph_signal_matrix_filename == 'pems_bay':
        path = 'data/pems_bay/pems_bay.h5'
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        mask_pre = ~np.isnan(df.values)

        mask_pre = np.expand_dims(mask_pre, 2)

        df.fillna(method='ffill', axis=0, inplace=True)

        data_seq1 = np.array(df.astype('float32'))

        data_seq = np.expand_dims(data_seq1, 2)


        eval_mask = pems_bay_mask_generator(data_seq.shape, p_fault=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4)

        mask = (eval_mask & mask_pre).astype('uint8')


        total_mask = mask

        # fitting first time
        y_hat = []

        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0

        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)

        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask = np.transpose(np.array(y_hat))

        # fitting second time
        SEED = 3215
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        new_mask = rand(data_seq_mask.shape) < 0.02

        for col in range(new_mask.shape[1]):
            idxs = np.flatnonzero(new_mask[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 12 * 4 > 12:
                fault_len = fault_len + int(randint(12 * 4 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, new_mask.shape[0] - 1)
            new_mask[idxs, col] = True
        new_mask = new_mask | (rand(new_mask.shape) < 0.05)

        mask1 = np.where(mask[:, :, 0] == 1, 0, 1)
        new_mask = mask1 & new_mask
        new_mask = new_mask.astype('uint8')

        y_hat = []
        new_mask[0] = 0
        new_mask[total_mask.shape[0] - 1] = 0

        for kk in range(new_mask.shape[1]):
            x = []
            y = []
            for ii in range(new_mask.shape[0]):
                if new_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        second_introplation_value = np.transpose(np.array(y_hat))

        # fitting third time
        SEED1 = 1999
        rng = np.random.default_rng(SEED1)
        rand = rng.random
        randint = rng.integers
        new_new_mask = rand(data_seq_mask.shape) < 0.02
        for col in range(new_new_mask.shape[1]):
            idxs = np.flatnonzero(new_new_mask[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 12 * 4 > 12:
                fault_len = fault_len + int(randint(12 * 4 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, new_new_mask.shape[0] - 1)
            new_new_mask[idxs, col] = True
        new_new_mask = new_new_mask | (rand(new_new_mask.shape) < 0.05)
        mask1 = np.where(mask[:, :, 0] == 1, 0, 1)
        new_new_mask = mask1 & new_new_mask
        new_new_mask = new_new_mask.astype('uint8')
        y_hat = []
        new_new_mask[0] = 0
        new_new_mask[total_mask.shape[0] - 1] = 0
        for kk in range(new_new_mask.shape[1]):
            x = []
            y = []
            for ii in range(new_new_mask.shape[0]):
                if new_new_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        third_introplation_value = np.transpose(np.array(y_hat))

        data_seq_mask = np.expand_dims(data_seq_mask, 2)


        second_introplation_value = np.expand_dims(second_introplation_value, 2)

        new_mask = np.expand_dims(new_mask, 2)

        third_introplation_value = np.expand_dims(third_introplation_value, 2)
        new_new_mask = np.expand_dims(new_new_mask, 2)

        all_samples = []

        # 第一次mask筛选
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, data_seq_mask, mask, mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train = []
        for idx in range(second_introplation_value.shape[0]):
            sample = get_sample_indices(data_seq, second_introplation_value, new_mask, new_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue
            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train2 = []
        for idx in range(second_introplation_value.shape[0]):
            sample = get_sample_indices(data_seq, third_introplation_value, new_new_mask, new_new_mask, num_of_weeks,
                                        num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue
            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train2.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

    if graph_signal_matrix_filename == 'AQI36':
        test_months = [3, 6, 9, 12]
        infer_eval_from = 'next'
        path = 'data/AQI/small36.h5'

        eval_mask = pd.DataFrame(pd.read_hdf(path, 'eval_mask'))
        eval_mask = eval_mask.values.astype('uint8')


        df = pd.DataFrame(pd.read_hdf(path, 'pm25'))

        mask = (np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0

        df = df.fillna(compute_mean(df))

        test_slice = np.in1d(df.index.month, test_months)
        train_slice = ~test_slice


        data_seq1 = np.array(df,dtype=float)


        # fitting original
        y_hat = []
        total_mask = mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq1 = np.transpose(np.array(y_hat))


        missing_mask= eval_mask

        # fitting first time
        y_hat = []
        total_mask = missing_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)
        data_seq_mask = np.transpose(np.array(y_hat))

        # fitting second time
        SEED = 56789
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask1 = rand(mask.shape) < 0.02
        for col in range(mask1.shape[1]):
            idxs = np.flatnonzero(mask1[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask.shape[0] - 1)
            mask1[idxs, col] = True
        mask1 = mask1 | (rand(mask1.shape) < 0.05)


        y_hat = []
        total_mask = mask1
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask1 = np.transpose(np.array(y_hat))

        # fitting third time
        SEED = 1999
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask2 = rand(mask.shape) < 0.015
        for col in range(mask2.shape[1]):
            idxs = np.flatnonzero(mask2[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask.shape[0] - 1)
            mask2[idxs, col] = True

        mask2 = mask2 | (rand(mask2.shape) < 0.05)


        y_hat = []
        total_mask = mask2
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask2 = np.transpose(np.array(y_hat))

        # fitting fourth time
        SEED = 54654
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask3 = rand(mask.shape) < 0.015
        for col in range(mask3.shape[1]):
            idxs = np.flatnonzero(mask3[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask3.shape[0] - 1)
            mask3[idxs, col] = True

        mask3 = mask3 | (rand(mask3.shape) < 0.05)

        y_hat = []
        total_mask = mask3
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])

            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask3 = np.transpose(np.array(y_hat))

        # fitting fifth time
        SEED = 34567
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask4 = rand(mask.shape) < 0.015
        for col in range(mask4.shape[1]):
            idxs = np.flatnonzero(mask4[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask4.shape[0] - 1)
            mask4[idxs, col] = True

        mask4 = mask4 | (rand(mask4.shape) < 0.05)

        y_hat = []
        total_mask = mask4
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask4 = np.transpose(np.array(y_hat))

        dataset = np.concatenate((data_seq1[train_slice],data_seq1[test_slice] ),axis=0)
        missing_mask = np.concatenate((eval_mask[train_slice],eval_mask[test_slice] ),axis=0)
        dataset = np.expand_dims(dataset, 2)
        missing_mask = np.expand_dims(missing_mask, 2)

        data_seq_mask = np.concatenate((data_seq_mask[train_slice],data_seq_mask[test_slice] ),axis=0)
        data_seq_mask1 = np.concatenate((data_seq_mask1[train_slice],data_seq_mask1[test_slice] ),axis=0)
        data_seq_mask2 = np.concatenate((data_seq_mask2[train_slice],data_seq_mask2[test_slice] ),axis=0)
        data_seq_mask3 = np.concatenate((data_seq_mask3[train_slice],data_seq_mask3[test_slice] ),axis=0)
        data_seq_mask4 = np.concatenate((data_seq_mask4[train_slice],data_seq_mask4[test_slice] ),axis=0)


        data_seq_mask = np.expand_dims(data_seq_mask, 2)
        data_seq_mask1 = np.expand_dims(data_seq_mask1, 2)
        data_seq_mask2 = np.expand_dims(data_seq_mask2, 2)
        data_seq_mask3 = np.expand_dims(data_seq_mask3, 2)
        data_seq_mask4 = np.expand_dims(data_seq_mask4, 2)

        all_samples = []

        for idx in range(dataset.shape[0]):
            sample = get_sample_indices(dataset,data_seq_mask, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train = []
        for idx in range(data_seq_mask1.shape[0]):
            sample = get_sample_indices(dataset,data_seq_mask1, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train2 = []
        for idx in range(data_seq_mask2.shape[0]):
            sample = get_sample_indices(dataset,data_seq_mask2, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train2.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train3 = []
        for idx in range(data_seq_mask3.shape[0]):
            sample = get_sample_indices(dataset,data_seq_mask3, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train3.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train4 = []
        for idx in range(data_seq_mask4.shape[0]):
            sample = get_sample_indices(dataset,data_seq_mask4, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train4.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

    if graph_signal_matrix_filename == 'AQI':
        test_months = [3, 6, 9, 12]
        infer_eval_from = 'next'
        path = 'data/AQI/full437.h5'
        eval_mask = None
        df = pd.DataFrame(pd.read_hdf(path, 'pm25'))

        mask = (np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0

        if eval_mask is None:
            eval_mask = infer_mask(df, infer_from=infer_eval_from)
        eval_mask = eval_mask.values.astype('uint8')  # 1 if value is ground-truth for imputation else 0

        df = df.fillna(compute_mean(df))

        test_slice = np.in1d(df.index.month, test_months)
        train_slice = ~test_slice

        data_seq1 = np.array(df)

        # fitting originally
        y_hat = []
        total_mask = mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq1 = np.transpose(np.array(y_hat))

        missing_mask = eval_mask

        # fitting first time
        y_hat = []
        total_mask = missing_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq1[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)
        data_seq_mask = np.transpose(np.array(y_hat))

        # fitting second time
        SEED = 56789
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask1 = rand(mask.shape) < 0.02
        for col in range(mask1.shape[1]):
            idxs = np.flatnonzero(mask1[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask.shape[0] - 1)
            mask1[idxs, col] = True
        mask1 = mask1 | (rand(mask1.shape) < 0.05)

        y_hat = []
        total_mask = mask1
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask1 = np.transpose(np.array(y_hat))

        # fitting third time
        SEED = 1999
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask2 = rand(mask.shape) < 0.015
        for col in range(mask2.shape[1]):
            idxs = np.flatnonzero(mask2[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask.shape[0] - 1)
            mask2[idxs, col] = True

        mask2 = mask2 | (rand(mask2.shape) < 0.05)

        y_hat = []
        total_mask = mask2
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask2 = np.transpose(np.array(y_hat))

        # fitting fourth time
        SEED = 54654
        rng = np.random.default_rng(SEED)
        rand = rng.random
        randint = rng.integers
        mask3 = rand(mask.shape) < 0.015
        for col in range(mask3.shape[1]):
            idxs = np.flatnonzero(mask3[:, col])
            if not len(idxs):
                continue
            fault_len = 12
            if 48 > 12:
                fault_len = fault_len + int(randint(48 - 12))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, mask3.shape[0] - 1)
            mask3[idxs, col] = True

        mask3 = mask3 | (rand(mask3.shape) < 0.05)

        y_hat = []
        total_mask = mask3
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(data_seq_mask[ii, kk])

            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq_mask3 = np.transpose(np.array(y_hat))


        dataset = np.concatenate((data_seq1[train_slice], data_seq1[test_slice]), axis=0)
        missing_mask = np.concatenate((eval_mask[train_slice], eval_mask[test_slice]), axis=0)
        dataset = np.expand_dims(dataset, 2)
        missing_mask = np.expand_dims(missing_mask, 2)

        data_seq_mask = np.concatenate((data_seq_mask[train_slice], data_seq_mask[test_slice]), axis=0)
        data_seq_mask1 = np.concatenate((data_seq_mask1[train_slice], data_seq_mask1[test_slice]), axis=0)
        data_seq_mask2 = np.concatenate((data_seq_mask2[train_slice], data_seq_mask2[test_slice]), axis=0)
        data_seq_mask3 = np.concatenate((data_seq_mask3[train_slice], data_seq_mask3[test_slice]), axis=0)

        data_seq_mask = np.expand_dims(data_seq_mask, 2)
        data_seq_mask1 = np.expand_dims(data_seq_mask1, 2)
        data_seq_mask2 = np.expand_dims(data_seq_mask2, 2)
        data_seq_mask3 = np.expand_dims(data_seq_mask3, 2)

        all_samples = []
        for idx in range(dataset.shape[0]):
            sample = get_sample_indices(dataset, data_seq_mask, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),

                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train = []
        for idx in range(data_seq_mask1.shape[0]):
            sample = get_sample_indices(dataset, data_seq_mask1, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train2 = []
        for idx in range(data_seq_mask2.shape[0]):
            sample = get_sample_indices(dataset, data_seq_mask2, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train2.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))

        all_samples_train3 = []
        for idx in range(data_seq_mask3.shape[0]):
            sample = get_sample_indices(dataset, data_seq_mask3, missing_mask, missing_mask, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, week_sample_mask, day_sample, day_sample_mask, hour_sample, hour_sample_mask, target, target_mask = sample
            all_samples_train3.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(week_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample_mask, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :],
                np.expand_dims(target_mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))


    if graph_signal_matrix_filename == 'AQI':
        split_line1 = int(len(all_samples) - 2 * 2928)
        split_line2 = int(len(all_samples) - 2928)

        training_set1 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples[:split_line2])]
        training_set2 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train[:split_line2])]
        training_set3 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train2[:split_line2])]
        training_set4 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train3[:split_line2])]


        training_set = []
        training_set.append(np.concatenate((training_set1[0], training_set2[0], training_set3[0], training_set4[0]), axis=0))
        training_set.append(np.concatenate((training_set1[1], training_set2[1], training_set3[1], training_set4[1]), axis=0))
        training_set.append(np.concatenate((training_set1[2], training_set2[2], training_set3[2], training_set4[2]), axis=0))
        training_set.append(np.concatenate((training_set1[3], training_set2[3], training_set3[3], training_set4[3]), axis=0))
        training_set.append(np.concatenate((training_set1[4], training_set2[4], training_set3[4], training_set4[4]), axis=0))
        training_set.append(np.concatenate((training_set1[5], training_set2[5], training_set3[5], training_set4[5]), axis=0))
        training_set.append(np.concatenate((training_set1[6], training_set2[6], training_set3[6], training_set4[6]), axis=0))
        training_set.append(np.concatenate((training_set1[7], training_set2[7], training_set3[7], training_set4[7]), axis=0))


    elif graph_signal_matrix_filename == 'AQI36':
        split_line1 = int(len(all_samples) - 2 * 2928)
        split_line2 = int(len(all_samples) - 2928)

        training_set1 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples[:split_line2])]
        training_set2 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train[:split_line2])]
        training_set3 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train2[:split_line2])]
        training_set4 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train3[:split_line2])]
        training_set5 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train4[:split_line2])]

        training_set = []
        training_set.append(np.concatenate((training_set1[0], training_set2[0], training_set3[0], training_set4[0], training_set5[0]), axis=0))
        training_set.append(np.concatenate((training_set1[1], training_set2[1], training_set3[1], training_set4[1], training_set5[1]), axis=0))
        training_set.append(np.concatenate((training_set1[2], training_set2[2], training_set3[2], training_set4[2], training_set5[2]), axis=0))
        training_set.append(np.concatenate((training_set1[3], training_set2[3], training_set3[3], training_set4[3], training_set5[3]), axis=0))
        training_set.append(np.concatenate((training_set1[4], training_set2[4], training_set3[4], training_set4[4], training_set5[4]), axis=0))
        training_set.append(np.concatenate((training_set1[5], training_set2[5], training_set3[5], training_set4[5], training_set5[5]), axis=0))
        training_set.append(np.concatenate((training_set1[6], training_set2[6], training_set3[6], training_set4[6], training_set5[6]), axis=0))
        training_set.append(np.concatenate((training_set1[7], training_set2[7], training_set3[7], training_set4[7], training_set5[7]), axis=0))



    elif graph_signal_matrix_filename == 'pems_bay':
        split_line1 = int(len(all_samples) - 2 * 10417)
        split_line2 = int(len(all_samples) - 10417)


        training_set1 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples[:split_line2])]
        training_set2 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train[:split_line2])]
        training_set3 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples_train2[:split_line2])]

        training_set = []
        training_set.append(np.concatenate((training_set1[0], training_set2[0], training_set3[0]), axis=0))
        training_set.append(np.concatenate((training_set1[1], training_set2[1], training_set3[1]), axis=0))
        training_set.append(np.concatenate((training_set1[2], training_set2[2], training_set3[2]), axis=0))
        training_set.append(np.concatenate((training_set1[3], training_set2[3], training_set3[3]), axis=0))
        training_set.append(np.concatenate((training_set1[4], training_set2[4], training_set3[4]), axis=0))
        training_set.append(np.concatenate((training_set1[5], training_set2[5], training_set3[5]), axis=0))
        training_set.append(np.concatenate((training_set1[6], training_set2[6], training_set3[6]), axis=0))
        training_set.append(np.concatenate((training_set1[7], training_set1[7], training_set1[7]), axis=0))
    elif graph_signal_matrix_filename == 'Electricity':
        split_line1 = int(len(all_samples) - 2 * 29183)
        split_line2 = int(len(all_samples) - 29183)
        training_set1 = [np.concatenate(i, axis=0)
                         for i in zip(*all_samples[:split_line2])]
        training_set = training_set1



    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_week,train_week_mask, train_day,train_day_mask, train_hour, train_hour_mask, train_target, train_mask = training_set
    val_week,val_week_mask, val_day, val_day_mask, val_hour,val_hour_mask, val_target, val_mask = validation_set
    test_week,test_week_mask, test_day,test_day_mask, test_hour,test_hour_mask, test_target, test_mask = testing_set

    print('training data: week: {},week_mask: {}, day: {}, day_mask: {},recent: {},recent_mask: {}, target: {}, mask: {}'.format(
        train_week.shape,train_week_mask.shape, train_day.shape,train_day_mask.shape,
        train_hour.shape,train_hour_mask.shape,  train_target.shape, train_mask.shape))
    print('validation data: week: {}, week_mask: {}, day: {},day_mask: {}, recent: {}, recent_mask: {},target: {}, mask: {}'.format(
        val_week.shape, val_week_mask.shape,val_day.shape, val_day_mask.shape,val_hour.shape, val_hour_mask.shape,val_target.shape, val_mask.shape))
    print('testing data: week: {}, week_mask: {}, day: {},day_mask: {}, recent: {}, recent_mask: {},target: {}, mask: {}'.format(
        test_week.shape,test_week_mask.shape, test_day.shape,test_day_mask.shape, test_hour.shape,test_hour_mask.shape, test_target.shape, test_mask.shape))

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    train_week_mask = (train_week_mask).transpose(0,2,1,3)
    train_day_mask = (train_day_mask).transpose(0,2,1,3)
    train_recent_mask = (train_hour_mask).transpose(0,2,1,3)



    val_week_mask = (val_week_mask).transpose(0,2,1,3)
    val_day_mask = (val_day_mask).transpose(0,2,1,3)
    val_recent_mask = (val_hour_mask).transpose(0,2,1,3)


    test_week_mask = (test_week_mask).transpose(0,2,1,3)
    test_day_mask = (test_day_mask).transpose(0,2,1,3)
    test_recent_mask = (test_hour_mask).transpose(0,2,1,3)


    all_data = {
        'train': {
            'week': train_week_norm,
            'week_mask': train_week_mask,
            'day': train_day_norm,
            'day_mask': train_day_mask,
            'recent': train_recent_norm,
            'recent_mask': train_recent_mask,
            'target': train_target,
            'target_mask': train_mask
        },
        'val': {
            'week': val_week_norm,
            'week_mask': val_week_mask,
            'day': val_day_norm,
            'day_mask': val_day_mask,
            'recent': val_recent_norm,
            'recent_mask': val_recent_mask,
            'target': val_target,
            'target_mask': val_mask
        },
        'test': {
            'week': test_week_norm,
            'week_mask': test_week_mask,
            'day': test_day_norm,
            'day_mask': test_day_mask,
            'recent': test_recent_norm,
            'recent_mask': test_recent_mask,
            'target': test_target,
            'target_mask': test_mask
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data

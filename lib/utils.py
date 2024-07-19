# -*- coding:utf-8 -*-
# pylint: disable=no-member

import csv
import numpy as np
import pandas as pd

from scipy.sparse.linalg import eigs

import torch


def pems_bay_mask_generator(shape,p_fault=0.0015, p_noise=0.05,min_seq=12,max_seq=12 * 4):
    SEED = 56789

    rng = np.random.default_rng(SEED)

    rand = rng.random
    randint = rng.integers
    mask = rand(shape) < p_fault
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True

    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


def Electricity_mask_generator(shape,p_fault=0.0015, p_noise=0.05,min_seq=12,max_seq=12 * 4):
    SEED = 9101112
    rng = np.random.default_rng(SEED)
    rand = rng.random
    randint = rng.integers
    mask = rand(shape) < p_fault
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True

    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')






def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1 if it is present in the DataFrame and
     absent in the `infer_from` month.
    @param pd.DataFrame df: the DataFrame.
    @param str infer_from: denotes from which month the evaluation value must be inferred.
    Can be either `previous` or `next`.
    @return: pd.DataFrame eval_mask: the evaluation mask for the DataFrame     """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns, data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('infer_from can only be one of %s' % ['previous', 'next'])
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        mask_j = mask[(mask.index.year == year_j) & (mask.index.month == month_j)]
        mask_i = mask_j.shift(1, pd.DateOffset(months=12 * (year_i - year_j) + (month_i - month_j)))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        eval_mask.loc[mask_i.index] = ~mask_i.loc[mask_i.index] & mask.loc[mask_i.index]
    return eval_mask


def compute_mean(x, index=None):
    """Compute the mean values for each datetime. The mean is first computed hourly over the week of the year.
     Further NaN values are computed using hourly mean over the same month through the years. If other NaN are present,
     they are removed using the mean of the sole hours. Hoping reasonably that there is at least a non-NaN entry of the
     same hour of the NaN datetime in all the dataset."""
    if isinstance(x, np.ndarray) and index is not None:
        shape = x.shape
        x = x.reshape((shape[0], -1))
        df_mean = pd.DataFrame(x, index=index)
    else:
        df_mean = x.copy()
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week, df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    units: int, week: 7 * 24, day: 24, recent(hour): 1

    points_per_hour: int, number of points per hour, depends on data

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence,data_seq_mask, total_mask, mask, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    if data_seq_mask.shape[1]==370:
        week_indices = search_data(data_seq_mask.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None

        day_indices = search_data(data_seq_mask.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None

        hour_indices = search_data(data_seq_mask.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None

        week_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in week_indices], axis=0)
        week_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in week_indices], axis=0)

        day_sample = np.concatenate([data_seq_mask[i: j]
                                     for i, j in day_indices], axis=0)
        day_sample_mask = np.concatenate([total_mask[i: j]
                                          for i, j in day_indices], axis=0)

        hour_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in hour_indices], axis=0)
        hour_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx - 3:  label_start_idx + 3]
        target_mask = mask[label_start_idx - 3: label_start_idx + 3]

    if data_seq_mask.shape[1]==325:
        week_indices = search_data(data_seq_mask.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None

        day_indices = search_data(data_seq_mask.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  1 * 24, points_per_hour)
        if not day_indices:
            return None

        hour_indices = search_data(data_seq_mask.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None

        week_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in week_indices], axis=0)
        week_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in week_indices], axis=0)

        day_sample = np.concatenate([data_seq_mask[i: j]
                                     for i, j in day_indices], axis=0)
        day_sample_mask = np.concatenate([total_mask[i: j]
                                          for i, j in day_indices], axis=0)

        hour_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in hour_indices], axis=0)
        hour_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx - 3:  label_start_idx + 3]
        target_mask = mask[label_start_idx - 3: label_start_idx + 3]


    if data_seq_mask.shape[1]==36:
        week_indices = search_data(data_seq_mask.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7*24, points_per_hour)
        if not week_indices:
            return None

        day_indices = search_data(data_seq_mask.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  1*24, points_per_hour)
        if not day_indices:
            return None

        hour_indices = search_data(data_seq_mask.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None

        week_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in week_indices], axis=0)
        week_sample_mask = np.concatenate([total_mask[i: j]
                                      for i, j in week_indices], axis=0)


        day_sample = np.concatenate([data_seq_mask[i: j]
                                     for i, j in day_indices], axis=0)
        day_sample_mask = np.concatenate([total_mask[i: j]
                                     for i, j in day_indices], axis=0)


        hour_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in hour_indices], axis=0)
        hour_sample_mask = np.concatenate([total_mask[i: j]
                                      for i, j in hour_indices], axis=0)



        target = data_sequence[label_start_idx - 3 :  label_start_idx +3]
        target_mask = mask[label_start_idx - 3 : label_start_idx +3]

    if data_seq_mask.shape[1] == 437:
        week_indices = search_data(data_seq_mask.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   1 * 24, points_per_hour)
        if not week_indices:
            return None

        day_indices = search_data(data_seq_mask.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None

        hour_indices = search_data(data_seq_mask.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None

        week_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in week_indices], axis=0)
        week_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in week_indices], axis=0)

        day_sample = np.concatenate([data_seq_mask[i: j]
                                     for i, j in day_indices], axis=0)
        day_sample_mask = np.concatenate([total_mask[i: j]
                                          for i, j in day_indices], axis=0)

        hour_sample = np.concatenate([data_seq_mask[i: j]
                                      for i, j in hour_indices], axis=0)
        hour_sample_mask = np.concatenate([total_mask[i: j]
                                           for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx - 3:  label_start_idx + 3]
        target_mask = mask[label_start_idx - 3: label_start_idx + 3]


    return week_sample,week_sample_mask, day_sample,day_sample_mask, hour_sample,hour_sample_mask, target, target_mask


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W
    
    lambda_max = eigs(L, k=1, which='LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, supports, device, epoch):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        tmp = []
        for index, (val_w,val_w_mask, val_d,val_d_mask, val_r, val_r_mask, val_t, val_t_mask) in enumerate(val_loader):
            val_w=val_w.to(device)
            val_w_mask = val_w_mask.to(device)

            val_d=val_d.to(device)
            val_d_mask=val_d_mask.to(device)

            val_r=val_r.to(device)
            val_r_mask=val_r_mask.to(device)

            val_t=val_t.to(device)
            val_t_mask= val_t_mask.to(device)



            output,_,_,ff = net(val_w, val_w_mask,val_d, val_d_mask, val_r,val_r_mask, val_t_mask, supports)
            l = loss_function(output * val_t_mask, val_t * val_t_mask)
            tmp.append(l.item())
    
        validation_loss = sum(tmp) / len(tmp)
    
        print('epoch: %s, validation loss: %.8f' % (epoch, validation_loss))
        return validation_loss


def predict(net, test_loader, supports, device):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    net.eval()
    with torch.no_grad():
        prediction = []
        dynamic_graph =[]
        for index, (test_w,test_w_mask, test_d, test_d_mask,test_r, test_r_mask, test_t, test_t_mask) in enumerate(test_loader):
            test_w=test_w.to(device)
            test_w_mask = test_w_mask.to(device)
            test_d=test_d.to(device)
            test_d_mask = test_d_mask.to(device)
            test_r=test_r.to(device)
            test_r_mask = test_r_mask.to(device)


            test_t=test_t.to(device)
            test_t_mask=test_t_mask.to(device)
            output,spatial_at,_,ff=net(test_w,test_w_mask, test_d, test_d_mask,test_r,test_r_mask,test_t_mask, supports)

            prediction.append(output.cpu().detach().numpy())
            dynamic_graph.append(spatial_at.cpu().detach().numpy())

        #get first batch's spatial attention matrix    
        for index, (test_w,test_w_mask, test_d, test_d_mask,test_r, test_r_mask, test_t, test_t_mask) in enumerate(test_loader):
            test_w=test_w.to(device)
            test_w_mask = test_w_mask.to(device)
            test_d=test_d.to(device)
            test_d_mask = test_d_mask.to(device)
            test_r=test_r.to(device)
            test_r_mask = test_r_mask.to(device)

            test_t=test_t.to(device)
            test_t_mask=test_t_mask.to(device)

            _,_,temporal_at,ff=net(test_w,test_w_mask, test_d, test_d_mask,test_r,test_r_mask,test_t_mask, supports)




            temporal_at=temporal_at.cpu().detach().numpy()
            break

        dynamic_graph = np.concatenate(dynamic_graph, 0)
        prediction = np.concatenate(prediction, 0)
        return prediction,dynamic_graph,temporal_at


def evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch):
    '''
    compute MAE, RMSE, MAPE scores of the prediction
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        prediction,_,_ = predict(net, test_loader, supports, device)

        mae1=[]
        mse1 =[]
        for i in range(6):
            err = np.abs(prediction[:, :, i] - true_value[:, :, i]) * true_value_mask[:, :, i]
            mae = err.sum() / true_value_mask[:, :, i].sum()

            err = np.square(prediction[:, :, i] - true_value[:, :, i]) * true_value_mask[:, :, i]
            mse = err.sum() / true_value_mask[:, :, i].sum()

            mae1.append(mae)
            mse1.append(mse)

        print('MAE: %.6f' % (min(mae1)))
        print('MSE: %.6f' % (min(mse1)))

        

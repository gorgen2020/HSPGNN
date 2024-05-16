# -*- coding: utf-8 -*-


import os
import shutil
from time import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset



parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 40]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.90, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=30, help='Size of temporal : 12')
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
parser.add_argument("--data_name", type=str, default='AQI36',
                    help="AQI,AQI36,pems_bay,Electricity", required=False)
parser.add_argument('--num_point', type=int, default=36,
                    help='road Point Number [437/36/325/370] ', required=False)
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
parser.add_argument('--model', type=str, default='HSPGCN' , help='HSPGCN or HSPGCN_L ')



FLAGS = parser.parse_args()



if FLAGS.model == 'HSPGCN':
    from model import HSPGCN as model
elif FLAGS.model == 'HSPGCN_L':
    from model import HSPGCN_L as model




decay=FLAGS.decay


num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
optimizer = FLAGS.optimizer
points_per_hour= 12
num_for_predict= 12
num_of_weeks = 1
num_of_days = 1
num_of_hours = 3
num_of_vertices=FLAGS.num_point

if FLAGS.data_name == 'pems_bay':
    graph_signal_matrix_filename = 'pems_bay'
    adj_filename = 'data/pems_bay/pems_bay_adj.npz'
    adj1 = np.load(adj_filename)['adj']
    adj1 = np.array(adj1 > 0.0, dtype=float)
    adj=np.matmul(adj1,adj1)
    adj = np.array(adj > 0.0, dtype=float)

    num_of_features=1
    batch_size = 8
    merge = False
elif FLAGS.data_name == 'Electricity':
    graph_signal_matrix_filename = 'Electricity'
    adj_filename = 'data/LA/LA_adj.npz'
    adj = np.ones([370,370])
    num_of_features = 1
    batch_size = 16
    merge = False
elif FLAGS.data_name == 'AQI':
    graph_signal_matrix_filename = 'AQI'
    adj_filename = 'data/AQI/AQI_adj.npz'
    adj1 = np.load(adj_filename)['adj']

    adj=np.matmul(adj1,adj1)
    adj = np.where(adj>0,1.0,0)

    num_of_features = 1
    batch_size = 16
    merge = False
elif FLAGS.data_name == 'AQI36':
    graph_signal_matrix_filename = 'AQI36'
    adj_filename = 'data/AQI/AQI_small_adj.npz'
    adj1 = np.load(adj_filename)['adj']

    adj = np.matmul(adj1, adj1)
    adj2 = np.where(adj1 > 0, adj1, adj)
    adj = np.matmul(adj2, adj1)
    adj = np.where(adj2 > 0, adj2, adj)

    num_of_features = 1
    batch_size = 8
    merge = False

if FLAGS.model == 'HSPGCN':
    model_name = 'HSPGCN_%s' % FLAGS.data_name
    params_dir = 'experiment_HSPGCN'
    prediction_path = 'HSPGCN_imputation_0%s' % FLAGS.data_name
elif FLAGS.model == 'HSPGCN_L':
    model_name = 'HSPGCN_L_%s' % FLAGS.data_name
    params_dir = 'experiment_HSPGCN_L'
    prediction_path = 'HSPGCN_L_imputation_0%s' % FLAGS.data_name





wdecay=0.000


device = torch.device(FLAGS.device)





supports=(torch.tensor(adj)).type(torch.float32).to(device)


print('Model is %s' % (model_name))

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))


if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    #Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)
    
    # test set ground truth
    true_value = all_data['test']['target']
    true_value_mask = all_data['test']['target_mask']

    print(true_value.shape)
    
    
    # training set data loader
    train_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['train']['week']),
                            torch.Tensor(all_data['train']['week_mask']),
                            torch.Tensor(all_data['train']['day']),
                            torch.Tensor(all_data['train']['day_mask']),
                            torch.Tensor(all_data['train']['recent']),
                            torch.Tensor(all_data['train']['recent_mask']),
                            torch.Tensor(all_data['train']['target']),
                            torch.Tensor(all_data['train']['target_mask'])
                        ),
                        batch_size=batch_size,
                        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['val']['week']),
                            torch.Tensor(all_data['val']['week_mask']),
                            torch.Tensor(all_data['val']['day']),
                            torch.Tensor(all_data['val']['day_mask']),
                            torch.Tensor(all_data['val']['recent']),
                            torch.Tensor(all_data['val']['recent_mask']),
                            torch.Tensor(all_data['val']['target']),
                            torch.Tensor(all_data['val']['target_mask'])
                        ),
                        batch_size=batch_size,
                        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['test']['week']),
                            torch.Tensor(all_data['test']['week_mask']),
                            torch.Tensor(all_data['test']['day']),
                            torch.Tensor(all_data['test']['day_mask']),
                            torch.Tensor(all_data['test']['recent']),
                            torch.Tensor(all_data['test']['recent_mask']),
                            torch.Tensor(all_data['test']['target']),
                            torch.Tensor(all_data['test']['target_mask'])
                        ),
                        batch_size=batch_size,
                        shuffle=False
    )
                        
    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )
    
    # loss function MSE
    loss_function = torch.nn.SmoothL1Loss(reduce=None, size_average=None,reduction='mean',beta=0.5)
    # loss_function = torch.nn.MSELoss()
    # get model's structure
    net = model(c_in=num_of_features,c_out=64,
                num_nodes=num_nodes,week=12,
                day=12,recent=36,
                K=3,Kt=3)
    net.to(device)#to cuda
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)


    # restore parameter
    # net.load_state_dict(torch.load('HSPGCN_L_pems_bay_epoch_15_0.09.params'))


    #calculate origin loss in epoch 0
    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)
    
    # compute testing set MAE, MSE, MAPE before training
    evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch=0)


    clip = 5
    his_loss =[]
    total_train_loss = []

    train_time=[]
    for epoch in range(1, epochs + 1):
        train_l=[]
        start_time_train = time()
        for train_w,train_w_mask, train_d,train_d_mask, train_r, train_r_mask, train_t, train_t_mask in train_loader:

            train_w=train_w.to(device)
            train_w_mask=train_w_mask.to(device)
            train_d=train_d.to(device)
            train_d_mask=train_d_mask.to(device)
            train_r=train_r.to(device)
            train_r_mask=train_r_mask.to(device)

            train_t=train_t.to(device)
            train_t_mask = train_t_mask.to(device)

            train_t_mask = 1-train_t_mask

            net.train() #train pattern
            optimizer.zero_grad() #grad to 0

            output,_,train_predict,ff = net(train_w,train_w_mask, train_d, train_d_mask,train_r,train_r_mask, train_t_mask, supports)

            ff=1

            loss = loss_function( (output * train_t_mask), (train_t * train_t_mask))

            #backward p
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            
            #update parameter
            optimizer.step()
            
            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l=np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, physics loss:%.8f, time: %.2fs'
                  % (epoch, train_l, ff, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)

        total_train_loss.append(train_l)

        # compute validation loss
        valid_loss=compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
        his_loss.append(valid_loss)
        
        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch)
        
        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                               epoch,str(round(valid_loss,2))))
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))
        
    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))
    
    
    bestid = np.argmin(his_loss)
        
    print("The valid loss on best model is epoch%s_%s"%(str(bestid+1), str(round(his_loss[bestid],4))))
    best_params_filename=os.path.join(params_path,
                                      '%s_epoch_%s_%s.params' % (model_name,
                                    str(bestid+1),str(round(his_loss[bestid],2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction,spatial_at,parameter_adj = predict(net, test_loader, supports, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch)
    test_time = np.mean(end_time_test-start_time_test)
    print("Test time: %.2f" % test_time)

    
    np.savez_compressed(
            os.path.normpath(prediction_path),
            adj=adj,
            prediction=prediction,
            spatial_at=spatial_at,
            parameter_adj=parameter_adj,
            ground_truth=all_data['test']['target'],
            ground_truth_mask=all_data['test']['target_mask']
        )
        
            
            
            
            
        
    
    
    
    
    
    
    
    

import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

from crowd_count import CrowdCounter
import network
from data_loader import ImageDataLoader
from timer import Timer
import utils
from evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = NoneNo

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def plot_training_curve():
    save_name = os.path.join(output_dir, 'training_curves.pkl')
    f = open(save_name, "rb")
    data = pickle.load(f)
    epoch_loss = data[0]
    epoch_mae = data[1]
    epoch_mse = data[2]
    epoch = range(len(epoch_loss))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(epoch, epoch_loss)
    ax1.set_title("Training Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss")
    ax2.plot(epoch, epoch_mae)
    ax2.set_title("Mean Absolute Error")
    ax2.set(xlabel="Epoch", ylabel="MAE")
    ax3.plot(epoch, epoch_mse)
    ax3.set_title("Mean Squared Error")
    ax3.set(xlabel="Epoch", ylabel="MSE")
    plt.autoscale(enable=True, axis="both")
    plt.show()

DEBUG = False
Print_training_curve = False
method = 'mcnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

if DEBUG:
    train_path = './data_debug/formatted_trainval/shanghaitech_part_A_patches_9/train'
    train_gt_path = './data_debug/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
    val_path = './data_debug/formatted_trainval/shanghaitech_part_A_patches_9/val'
    val_gt_path = './data_debug/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

if Print_training_curve:
    plot_training_curve()

#training configuration
start_step = 0
end_step = 5
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250


#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
epoch_loss = []
epoch_mae = []
epoch_mse = []
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = 9223372036854775807

for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0
    for blob in data_loader:                
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.data.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % disp_interval == 0:            
            duration = t.toc(average=False)
            t.tic()
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)    
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                step, duration, gt_count,et_count)
            utils.save_results(im_data,gt_data,density_map, output_dir)

            log_print(log_text, color='green', attrs=['bold'])

    # Append train_loss
    epoch_loss.append(train_loss)



    if (epoch % 10 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net)

        # calculate error on the validation dataset
        mae, mse = evaluate_model(save_name, data_loader_val)
        # Append MAE MSE
        epoch_mae.append(mae)
        epoch_mse.append(mse)
        # save trainning curves
        save_name = os.path.join(output_dir, 'training_curves.pkl')
        f = open(save_name, "wb")
        pickle.dump([epoch_loss, epoch_mae, epoch_mse], f)
        f.close()

        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)


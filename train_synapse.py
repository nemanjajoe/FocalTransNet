# ------------------------------------------
# DAFN: Dual Attention Fusion Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import math
import os
import random
from matplotlib.widgets import EllipseSelector
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import datetime
import logging
import sys
import json
import argparse

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from networks.dataset import Synapse_dataset
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from config import config_synapse
from thop import profile,clever_format
from tqdm import tqdm

def inference(model, hyper, labels, save_infrence=False, test_mode=False):
   model.eval()
   metric_list = 0.0
   num_classes = hyper['num_classes']
   img_size = hyper['model_args']['img_size']
   test_save_path = None
   if save_infrence:
      test_save_path = os.path.join(hyper['save_path'],'inference')
      if os.path.exists(test_save_path) is False:
         os.makedirs(test_save_path)

   device = torch.device(hyper['device'])
   dataset_path = hyper['dataset_path'] 
   test_loader_args = hyper['test_loader_args'] if 'test_loader_args' in hyper else {}
   test_path = os.path.join(dataset_path,'test_vol_h5')
   test_dataset = Synapse_dataset(base_dir=test_path, split="test_vol", list_dir=dataset_path, img_size=img_size,shuffle=False , test_mode=test_mode)
   test_loader = DataLoader(test_dataset, **test_loader_args)
   length = len(test_loader)

   for i_batch,sample in enumerate(test_loader):
      image, label, case_name = sample['image'], sample['label'], sample['case_name'][0]
      metric_i = test_single_volume(image, label, model, device, classes=num_classes, 
                                    patch_size=[img_size, img_size], test_save_path=test_save_path, case=case_name)
      metric_list += np.array(metric_i)
      metric_mean = np.mean(metric_i,axis=0)
      logging.info(f'\t\t {case_name}: mean dice {metric_mean[0]:.4f}, mean hd95 {metric_mean[1]:.4f}')

   metric_list /= length
   
   # labels = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
   for i, organ in enumerate(labels):
            logging.info(f'\t\t {organ} : dice {metric_list[i][0]:.4f}, hd95 {metric_list[i][1]:.4f}') # type: ignore
   

   logging.info(f'\t\t mean dice {np.mean(metric_list,axis=0)[0]:.4f}, mean hd95 {np.mean(metric_list,axis=0)[1]:.4f}')

   return np.asarray(metric_list) #[[dice][hd95]]

def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def train_epoch(model, criterion, optimizer,scheduler, hyper, test_mode=False, train_loader=None):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    grad_clipping = hyper['grad_clipping']
    device = torch.device(hyper['device'])
    lr_ = optimizer.param_groups[-1]['lr']

    if train_loader is None:
       dataset_path = hyper['dataset_path']
       img_size = hyper['model_args']['img_size'] if 'img_size' in hyper['model_args'].keys() else 224
       train_loader_args = hyper['train_loader_args'] if 'train_loader_args' in hyper else {}
       x_transforms = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize([0.5], [0.5])
       ])
       y_transforms = transforms.ToTensor()
   
       train_path = os.path.join(dataset_path,'train_npz')
       train_dataset = Synapse_dataset(base_dir=train_path, list_dir=dataset_path, split="train",img_size=img_size,
                                  norm_x_transform = x_transforms, norm_y_transform = y_transforms,shuffle=True,test_mode=test_mode)
       train_loader = DataLoader(train_dataset, **train_loader_args)

    length = len(train_loader)
    t_start = time.time()
    batch_size = 1
    for sample in train_loader:
      x,y = sample['image'], sample['label']
      batch_size = x.shape[0]
      x = x.to(device)
      y = y.to(device).squeeze(1)
      
      y_ = model(x)

      loss,ce,dice = criterion(y_, y)
      running_loss += loss.detach().cpu().item()
      running_ce += ce.detach().cpu().item()
      running_dice += dice.detach().cpu().item()

      optimizer.zero_grad()
      loss.backward()
      if grad_clipping:
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        
      optimizer.step()
      if scheduler is not None:
         scheduler.step()
         lr_ = scheduler.get_last_lr()[-1]
      # else:
      #    lr_ = args.base_lr * (1.0 - args.iter_num / args.max_iterations) ** 0.9
      #    for param_group in optimizer.param_groups:
      #       param_group['lr'] = lr_
      
      # args.iter_num += 1
      # logging.info('iteration %d : lr: %f, loss : %f, loss_ce: %f, loss_dice: %f' % (args.iter_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
    
    interval = time.time() - t_start
    throughput = (batch_size*length)/interval
    logging.info(f'\t\t loss:{running_loss/length:.4f}')
    logging.info(f'\t\t ce loss   :{running_ce/length:.4f}')
    logging.info(f'\t\t dice loss :{running_dice/length:.4f}')
    logging.info(f'\t\t steps:{length}')
    logging.info(f'\t\t throughput:{throughput:.2f} ')
    logging.info(f'\t\t tps :{batch_size/throughput:.2f}')
    logging.info(f'\t\t lr :{lr_}')
    # return running_loss/length, running_ce/length, running_dice/length, lr_


def _save_best(model,optimizer,hyper,save_epoch_path,epoch, best_metric, labels):
   dice_best = list(best_metric[:,0])
   hd95_best = list(best_metric[:,1])
   params = {
     'epoch' : epoch,
     'state_dict' : model.state_dict(),
     'optimizer_state_dict' : optimizer.state_dict(),
     'dice_list' : dice_best,
     'hd95_list' : hd95_best,
     'labels'    : labels,
     'hyper': hyper
   }
   torch.save(params, save_epoch_path)
   logging.info(f'\t save best epoch success!')


def train(model, criterion, optimizer,scheduler, hyper,labels,continue_train=False, epoch=None):
   num_classes = hyper['num_classes']
   epoch_num = hyper['epoch_num']
   save_path = hyper['save_path']
   save_epoch_path = os.path.join(save_path, 'best_epoch.pth')
   eval_frequncy = hyper['eval_frequncy']

   labels = [label for label in labels.values()][:num_classes]
   if epoch is None:
      epoch_num = epoch_num
   else:
      epoch_num = epoch
   epoch_start = 0
   best_metric = np.zeros((8,2),dtype=np.float32)
   if continue_train:
     if os.path.exists(save_epoch_path):
       params = torch.load(save_epoch_path)
       dice_list = params['dice_list']
       hd95_list = params['hd95_list']
       best_metric = np.asarray([dice_list,hd95_list]).transpose()
       epoch_start = params['epoch'] + 1
       logging.info(f'continue training from best epoch {epoch_start}')
       for i, organ in enumerate(params['labels']):
         logging.info(f'\t {organ} : dice={dice_list[i]:.4f},hd95={hd95_list[i]:.4f}')
       model.load_state_dict(params['state_dict'])
       optimizer.load_state_dict(params['optimizer_state_dict'])
       del params
     else:
        params = torch.load(os.path.join(save_path,"epoch.pth"))
        epoch_start = params['epoch'] + 1
        logging.info(f'continue training from epoch {epoch_start}')
        model.load_state_dict(params['state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        del params
   
   logging.info("---------- start trainning ----------")
   for epoch in range(epoch_start, epoch_num):
       t_tick = time.time()
       logging.info(f'epoch {epoch+1}/{epoch_num}:')
       logging.info(f'\t train:')
       train_epoch(model=model, criterion=criterion, optimizer=optimizer,scheduler=scheduler,
                   hyper=hyper,test_mode=False)
       if (epoch + 1)%hyper['save_frequncy'] == 0:
         params = {
                   'epoch' : epoch,
                   'state_dict' : model.state_dict(),
                   'optimizer_state_dict' : optimizer.state_dict()
                   }
         torch.save(params, os.path.join(save_path,"epoch.pth"))
         del params

       do_inference = (epoch+1) % eval_frequncy == 0 or epoch == epoch_num-1
       if do_inference:
         logging.info('\t inference:')
         metric = inference(model=model, hyper=hyper,labels=labels[1:],
                            save_infrence=False, test_mode=False)
         if np.mean(metric[:,0]) >= np.mean(best_metric[:,0]) or np.mean(metric[:,1]) < np.mean(metric[:,1]):
           best_metric = metric
           _save_best(model,optimizer,hyper,save_epoch_path,epoch,best_metric,labels[1:])
       logging.info(f'\t time: {(time.time() - t_tick):.2f} s')
   result_data = np.asarray(best_metric)
   result = pd.DataFrame(result_data,index=labels[1:],columns=['dice','hd95'])
   date = time.strftime("%Y%m%d_%H%M",time.localtime(time.time()))
   result.to_csv(os.path.join(save_path,f"test_result_{date}.csv"))


def train_synapse(continue_train=None, plot=False, hyper=None):  
   hyper = hyper or config_synapse

   if args.data_path is not None:
      hyper['dataset_path'] = os.path.abspath(args.data_path)
   
   if args.save_path is not None:
      hyper['save_path'] = os.path.abspath(args.save_path)
   #---------- save path configurations ----------
   save_path = hyper['save_path']
   if os.path.exists(save_path) == False:
      os.makedirs(save_path)
   log_path = os.path.join(save_path, 'log.txt')

   logging.basicConfig(filename=log_path, level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
   logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
   
   logging.info('Synapse Trainer Initailizing...')
   logging.info(f'result will be saved on {save_path}')
   save_epoch_path = os.path.join(save_path, 'best_epoch.pth')
   
   if os.path.exists(save_epoch_path) or os.path.exists(os.path.join(save_path,'epoch.pth')):
      logging.info("========================warning====================")
      logging.info("This config was trained before, continue training will cover the saved data!")
      time.sleep(1)
      logging.info("====================================================")
     # date = time.strftime("%Y%m%d_%H%M",time.localtime(time.time()))
     # os.rename(save_epoch_path,os.path.join(save_path,f'best_epoch_{date}.pth'))

   #---------- setup model ----------
   
   device = torch.device(hyper['device'])
   model = hyper['model'](**hyper['model_args']).to(device)
   if hyper['pretrained_params'] is not None:
     model.load_from(hyper['pretrained_params'], device)
     model.to(device)
   criterion = hyper['criterion'](**hyper['criterion_args'])
   optimizer = hyper['optimizer'](model.parameters(),**hyper['optimizer_args'])
   scheduler = hyper['scheduler'](optimizer,**hyper['scheduler_args']) if 'scheduler' in hyper else None
   
   #---------- relavent args ----------
   if hyper['n_gpu'] > 1:
       model = nn.DataParallel(model)
   labels = {0: 'background',  1: 'aorta',      2: 'gallbladder',
             3: 'left kidney', 4: 'right kidney', 5: 'liver',
             6: 'pancreas',     7: 'spleen',       8: 'stomach'}
#    labels = {0:'background',1:'spleen',2:'right kidney',3:'left kidney',
# 4:'gallbladder',5:'liver',6:'stomach',7:'aorta',8:'pancreas'}

   logging.info('Synapse Trainer initalied !')
   if continue_train is not None:
      train(model,criterion,optimizer,scheduler,hyper,labels,True,continue_train)
   
   if continue_train is None and plot is False: # training from scratch
      train(model,criterion,optimizer,scheduler,hyper,labels,False)
   
   # plot multi organ segmentation map
   if os.path.exists(save_epoch_path):
      epoch = torch.load(save_epoch_path)
      model.load_state_dict(epoch['state_dict'])
      del epoch
   else:
      print('model is not trained, can not plot !')
      return

   labels = [label for label in labels.values()][:hyper['num_classes']]
   logging.info(f'plot seg map')
   inference(model=model, hyper=hyper,labels=labels[1:],save_infrence=True)


def print_model_size(model):
    x = torch.randn((1,1,224,224))
    flops,params = profile(model=model, inputs=(x,)) # type: ignore
    flops,params = clever_format([flops,params], "%.3f")
    print(flops,params)

parser = argparse.ArgumentParser()
parser.add_argument('--continue_train', type=int,
                    default=None, help='continue train from last epoch')
parser.add_argument('--test', action='store_true', help='evaluate model on test dataset')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--data-path', type=str, help='dataset path',
                    default=None)
parser.add_argument('--save-path', type=str, help='results saving path',
                    default=None)
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_synapse(args.continue_train, args.test)
    
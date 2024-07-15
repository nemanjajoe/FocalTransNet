# ------------------------------------------
# DAFN: Dual Attention Fusion Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import logging
import sys
import argparse
import torchmetrics
import cv2 as cv

from torch.utils.data import DataLoader
from networks.dataset import SegPC2021_dataset
from torchvision import transforms
from config import config_SegPC2021
from thop import profile,clever_format
from tqdm import tqdm

def inference(model, hyper, metrics):
   model.eval()
   img_size = hyper['model_args']['img_size']

   device = torch.device(hyper['device'])
   dataset_path = hyper['dataset_path']
   test_loader_args = hyper['test_loader_args'] if 'test_loader_args' in hyper else {}
   x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
   y_transforms = transforms.ToTensor()
   test_dataset = SegPC2021_dataset(base_dir=dataset_path, split="test_npz", img_size=img_size,
                                   norm_x_transform=x_transforms,norm_y_transform=y_transforms,shuffle=False)
   test_loader = DataLoader(test_dataset, **test_loader_args)
   evaluator = metrics.clone().to(device)
   test_progress = tqdm(test_loader,dynamic_ncols=True,leave=False)
   with torch.no_grad():
      for sample in test_progress:
         x, y, img_name = sample['image'], sample['label'], sample['name'][0]
         assert(x.shape[0] == 1)
         x = x.to(device) # B 4 224 224
         y = y.to(device).squeeze(1) # B 224 224
         
         y_ = model(x)
         
         not_nucs = torch.where(x[:,-1,:,:]>0, 0, 1)
         preds_ = torch.argmax(y_,1,keepdim=False).float() * not_nucs
         mask = torch.argmax(y, 1, keepdim=False) * not_nucs
         evaluator.update(preds_,mask.int())
         
   result = evaluator.cpu().compute()
   total = 0.
   for k in result.keys():
      m = result[k] = result[k].item()
      total += m
      logging.info(f'\t\t {k} : {result[k]:.4f}')
   
   mean = total/len(result.keys())
   logging.info(f'\t\t mean metric:{mean}')
   return result,mean

def train_epoch(model, criterion, optimizer,scheduler, hyper, metrics):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    grad_clipping = hyper['grad_clipping']
    device = torch.device(hyper['device'])
    lr_ = optimizer.param_groups[-1]['lr']
    evaluator = metrics.clone().to(device)
    dataset_path = hyper['dataset_path']
    img_size = hyper['model_args']['img_size'] if 'img_size' in hyper['model_args'].keys() else 224
    train_loader_args = hyper['train_loader_args'] if 'train_loader_args' in hyper else {}
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()
    train_dataset = SegPC2021_dataset(base_dir=dataset_path, split="train_npz",img_size=img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms,shuffle=True,test_mode=False)
    train_loader = DataLoader(train_dataset, **train_loader_args)
    
    length = len(train_loader)
   #  train_progress = tqdm(train_loader,leave=False)
    for sample in tqdm(train_loader,total=length,dynamic_ncols=True,leave=False):
      x,y = sample['image'], sample['label']
      x = x.to(device)
      y = y.to(device)
   
      y_ = model(x)

      loss,ce,dice = criterion(y_, y)
      running_loss += loss
      running_ce += ce
      running_dice += dice
      # print(f'sample {idx}: loss: {loss:.4f}, ce:{ce:.4f}, dice:{dice:.4f} ')

      optimizer.zero_grad()
      loss.backward()
      if grad_clipping:
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        
      optimizer.step()
      if scheduler is not None:
         scheduler.step()
         lr_ = scheduler.get_last_lr()[-1]

      not_nucs = torch.where(x[:,-1,:,:]>0, 0, 1)
      preds_ = torch.argmax(y_,1,keepdim=False).float() * not_nucs
      mask = torch.argmax(y, 1, keepdim=False) * not_nucs
      evaluator.update(preds_,mask.int())
         
    logging.info(f'\t\t loss:{running_loss/length:.4f}')
    logging.info(f'\t\t ce loss   :{running_ce/length:.4f}')
    logging.info(f'\t\t dice loss :{running_dice/length:.4f}')
    logging.info(f'\t\t lr :{lr_}')

    result = evaluator.cpu().compute()
    total = 0.
    for k in result.keys():
       m = result[k] = result[k].item()
       total += m
       logging.info(f'\t\t {k} : {result[k]:.4f}')
    
    mean = total/len(result.keys())
    logging.info(f'\t\t mean metric:{mean}')
    return result,mean

def _save_best(model,optimizer,hyper,save_epoch_path,epoch, best_metric):
   params = {
     'epoch' : epoch,
     'state_dict' : model.state_dict(),
     'optimizer_state_dict' : optimizer.state_dict(),
     'metric' : best_metric,
     'hyper': hyper
   }
   torch.save(params, save_epoch_path)
   logging.info(f'\t save best epoch success!')


def train(model, criterion, optimizer,scheduler, hyper,continue_train=False, epoch=None):
  epoch_num = hyper['epoch_num']
  save_path = hyper['save_path']
  save_best_epoch_path = os.path.join(save_path, 'best_epoch.pth')
  eval_frequncy = hyper['eval_frequncy']
  metrics = torchmetrics.MetricCollection([   
      #   torchmetrics.F1Score(),
        torchmetrics.Accuracy(),
        torchmetrics.Dice(),
        torchmetrics.Precision(),
        torchmetrics.Specificity(),
        torchmetrics.Recall(),
        # IoU
        torchmetrics.JaccardIndex(2)
    ])
  train_metrics = metrics.clone(prefix='train_metric/')
  test_metrics = metrics.clone(prefix='inference_metric/')
#   train_metrics = metrics.clone(prefix='train_metrics/')

  if epoch is None:
     epoch_num = epoch_num
  else:
     epoch_num = epoch
  epoch_start = 0
  if continue_train:
    if os.path.exists(save_best_epoch_path):
      params = torch.load(save_best_epoch_path)
      epoch_start = params['epoch']
      logging.info(f'continue training from best epoch {epoch_start}')
      model.load_state_dict(params['state_dict'])
      optimizer.load_state_dict(params['optimizer_state_dict'])
      del params
    else:
       params = torch.load(os.path.join(save_path,"epoch.pth"))
       epoch_start = params['epoch']
       logging.info(f'continue training from epoch {epoch_start}')
       model.load_state_dict(params['state_dict'])
       optimizer.load_state_dict(params['optimizer_state_dict'])
       del params
  
  logging.info("---------- start trainning ----------")
  
  
  best_metric = 0.
  train_history_results = test_history_results = pd.DataFrame()
  for epoch in range(epoch_start, epoch_num):
      t_tick = time.time()
      logging.info(f'epoch {epoch+1}/{epoch_num}:')
      logging.info(f'\t train:')
      train_results,_ = train_epoch(model=model, criterion=criterion, optimizer=optimizer,scheduler=scheduler,
                  hyper=hyper,metrics=train_metrics)
      
      train_results = pd.DataFrame(train_results,index=[epoch])
      train_history_results = pd.concat([train_history_results,train_results])
      if (epoch + 1)%hyper['save_frequncy'] == 0:
        params = {
                  'epoch' : epoch,
                  'state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict()
                  }
        torch.save(params, os.path.join(save_path,"epoch.pth"))
        logging.info(f'\t save epoch success !')
        del params
      
      do_inference = (epoch >= 0 and (epoch+1) % eval_frequncy == 0) or epoch == epoch_num-1
      # do_inference = True
      if do_inference:
         logging.info('\t inference:')
         test_results,mean_metric = inference(model=model, hyper=hyper,metrics=test_metrics)
         
         test_history_results = pd.concat([test_history_results,pd.DataFrame(test_results,index=[epoch])])
         if mean_metric > best_metric:
            best_metric = mean_metric
            # date = time.strftime("%Y%m%d_%H%M",time.localtime(time.time()))
            pd.Series(test_results).to_csv(os.path.join(save_path,f'result.csv'))
            _save_best(model,optimizer,hyper,save_best_epoch_path,epoch,best_metric)
      
      logging.info(f'\t time: {(time.time() - t_tick):.2f} s')


  # save history metrics results
  train_history_results.to_csv(os.path.join(save_path,'train_history.csv'))
  test_history_results.to_csv(os.path.join(save_path,'test_history.csv'))
  

def train_isic(continue_train=None, test= False, hyper=None):  
   hyper = hyper or config_SegPC2021

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
   logging.info('Synapse Trainer initalied !')
   
   if continue_train is not None: # training from checkpoint
      train(model,criterion,optimizer,scheduler,hyper,True,continue_train)
   
   if continue_train is None and test is False: # training from scratch
      train(model,criterion,optimizer,scheduler,hyper)
   
   # plot seg map
   logging.info(f'start ploting...')
   if os.path.exists(save_epoch_path):
      epoch = torch.load(save_epoch_path)
      model.load_state_dict(epoch['state_dict'])
      del epoch
   else:
      print('model is not trained, can not plot!')
      return
 
   test_loader_args = {'batch_size':1, 'shuffle':False}
   img_size = hyper['model_args']['img_size']
   x_transform = transforms.Compose([
         # transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
     ])
   test_dataset = SegPC2021_dataset(base_dir=hyper['dataset_path'], split="test_npz", img_size=img_size,shuffle=False,one_hot=False)
   test_loader = DataLoader(test_dataset, **test_loader_args)
   test_save_path = os.path.join(hyper['save_path'],'inference')
   
   if os.path.exists(test_save_path) is False:
      os.makedirs(test_save_path)
   
   colors = np.asarray([
      [0,0,255],[255,0,0]
   ]).astype(np.float32)
   colors = colors/255.
   with torch.no_grad():
      for sample in tqdm(test_loader):
         img,y,name = sample['image'],sample['label'],sample['name'][0]

         x = x_transform(img.type(torch.float32).permute(0,3,1,2)).to(device)
         y_ = model(x).cpu()
         preds_ = torch.argmax(y_,1,keepdim=False).float()
         pred = np.float32(preds_.squeeze(0).numpy())
         gt = np.float32(y.squeeze(0).squeeze(0).numpy())
         img = img.squeeze(0).numpy()
         raw_img = cv.cvtColor(img[:,:,:3],cv.COLOR_RGB2BGR)
         nucl_msk = np.float32(img[:,:,-1])
         pred = pred*np.where(nucl_msk == 1,0,1)
         pred = np.where(pred == 0, 0, 1)
         pred_color = np.zeros(raw_img.shape,np.float32)
         gt_color = np.zeros(raw_img.shape,np.float32)
         bg_red = np.ones(raw_img.shape,np.float32)*colors[1]
         bg_blue = np.ones(raw_img.shape,np.float32)*colors[0]
         for i in range(3):
            pred_color[:,:,i] = bg_red[:,:,i]*pred + bg_blue[:,:,i]*nucl_msk
            gt_color[:,:,i] = bg_red[:,:,i]*gt + bg_blue[:,:,i]*nucl_msk

         pred_color = cv.normalize(pred_color, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC3)
         gt_color = cv.normalize(gt_color, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC3)
         
         cv.imwrite(os.path.join(test_save_path,name + '_raw_img.jpg'),raw_img)
         cv.imwrite(os.path.join(test_save_path,name + '_pred.jpg'),pred_color)
         cv.imwrite(os.path.join(test_save_path,name + '_gt.jpg'),gt_color)

def print_model_size(net):
    x = torch.randn((1,1,224,224)).to(net.device)
    flops,params = profile(model=net.model, inputs=(x,)) # type: ignore
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

    train_isic(args.continue_train, args.test)
    
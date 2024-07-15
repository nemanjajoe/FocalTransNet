import torch
import os

from networks import Net as FocalTransNet
from utils import SynapseLoss,ISICLoss,SegPCLoss

data_dir = r'./data'
save_dir = r'./results'

NCLS=4
config_PDGM= {
  'describe'  : "PDGM dataset", 
  'save_path' : os.path.join(save_dir,'PDGM'),
  'dataset_path': os.path.join(data_dir,'PDGM'),
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 20,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FocalTransNet,
  'model_args': {'img_size':224, 'dim_in':8, 'dim_out':NCLS,'scale_factor':2,'embed_dim':96,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'depth':[1,2,4,2]},
  'criterion' : SynapseLoss,
  'criterion_args': {'n_classes' : NCLS, 'alpha':0.1, 'beta':0.9}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.008},
  'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
  'scheduler_args':{'T_max':90000},
  'train_loader_args' : {'batch_size':6, 'shuffle':True, 'num_workers':6, 'pin_memory':True, 'drop_last':False},
  'test_loader_args': {'batch_size':1, 'shuffle':False, 'num_workers':1},
  'eval_frequncy':1, # test/inference frequncy
  'save_frequncy':999,  # check-point frequncy
  'n_gpu': 1,
  'grad_clipping': False,
}

NCLS=9
config_synapse= {
  'describe'  : "Synapse dataset", 
  'save_path' : os.path.join(save_dir,'Synapse'),
  'dataset_path': os.path.join(data_dir,'Synapse'),
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 180,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FocalTransNet,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':NCLS,'scale_factor':2,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'depth':[1,2,4,2]},
  'criterion' : SynapseLoss,
  'criterion_args': {'n_classes' : NCLS, 'alpha':0.1, 'beta':0.9}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.008},
  'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
  'scheduler_args':{'T_max':90000},
  'train_loader_args' : {'batch_size':6, 'shuffle':True, 'num_workers':6, 'pin_memory':True, 'drop_last':False},
  'test_loader_args': {'batch_size':1, 'shuffle':False, 'num_workers':1},
  'eval_frequncy':5, # test/inference frequncy
  'save_frequncy':5,  # check-point frequncy
  'n_gpu': 1,
  'grad_clipping': False,
}

NCLS = 2
config_ISIC2018= {
  'describe'  : "ISIC 2018 dataset", 
  'save_path' : os.path.join(save_dir,'ISIC2018'),
  'dataset_path': os.path.join(data_dir,'ISIC2018'),
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 100,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FocalTransNet,
  'model_args': {'img_size':224, 'dim_in':3, 'dim_out':NCLS,'scale_factor':2,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'depth':[1,2,4,2]},
  'criterion' : ISICLoss,
  'criterion_args': {'n_classes' : NCLS, 'alpha':0.8, 'beta':0.2}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.03},
  'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
  'scheduler_args':{'T_max':20000},
  'train_loader_args' : {'batch_size':6, 'shuffle':True, 'num_workers':6, 'pin_memory':True, 'drop_last':False},
  'test_loader_args': {'batch_size':1, 'shuffle':False, 'num_workers':1},
  'eval_frequncy':1, # test/inference frequncy
  'save_frequncy':999,  # check-point frequncy
  'n_gpu': 1,
  'grad_clipping': False,
}

config_SegPC2021= {
  'describe'  : "SegPC2021 dataset", 
  'save_path' : os.path.join(save_dir,'SegPC2021'),
  'dataset_path': os.path.join(data_dir,'SegPC2021'),
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 80,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FocalTransNet,
  'model_args': {'img_size':224, 'dim_in':4, 'dim_out':NCLS,'scale_factor':2,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'depth':[1,2,4,2]},
  'criterion' : SegPCLoss,
  'criterion_args': {'n_classes' : NCLS, 'alpha':0.8, 'beta':0.2}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.03},
  'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
  'scheduler_args':{'T_max':20000},
  'train_loader_args' : {'batch_size':6, 'shuffle':True, 'num_workers':6, 'pin_memory':True, 'drop_last':False},
  'test_loader_args': {'batch_size':1, 'shuffle':False, 'num_workers':1},
  'eval_frequncy':1, # test/inference frequncy
  'save_frequncy':999,  # check-point frequncy
  'n_gpu': 1,
  'grad_clipping': False,
}

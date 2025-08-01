# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
import time
import cv2
from sam2_train.modeling.utils_sr.matlab_resize import imresize
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader
from sam2_train.modeling.diffusion_sam import GaussianDiffusion_sam
import cfg
import func_2d.function as function
from conf import settings
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *
from cfg import hparams
from tqdm import tqdm
from sam2_train.modeling.diffsr_module import RRDBNet, Unet
from sam2_train.modeling.utils_sr.utils import load_ckpt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sam2_train.modeling.utils_sr.dataset import SRDataSet
from sam2_train.modeling.utils_sr.indexed_datasets import IndexedDataset



from sam2_train.modeling.utils_sr.utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars, Measure, \
    get_all_ckpts
    
    
    
def save_checkpoint(model, optimizer, work_dir, global_step):
  
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')

    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}


    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)


def build_scheduler(optimizer):
        if 'scheduler' in hparams:
            scheduler_config = hparams['scheduler']
            if scheduler_config['type'] == 'cosine':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hparams['max_updates'],
                                                                          eta_min=scheduler_config['eta_min'])
        
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)
        
        return lr_scheduler     
  

 
def test():

     # use bfloat16 for the entire work
    
   
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
 
    
    args = cfg.parse_args()
    print(hparams)
    #set_hparams()
    #dist.init_process_group(backend="nccl")
   # torch.cuda.set_device(args.local_rank)
    #device = torch.device("cuda", args.local_rank)
    #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    GPUdevice = torch.device('cuda', args.gpu_device)
 
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    

    if args.SR == True:
    
       
          hidden_size = hparams['hidden_size']
          dim_mults = hparams['unet_dim_mults']
          dim_mults = [int(x) for x in dim_mults.split('|')]
        
          denoise_fn = Unet(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
          if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
          else:
              rrdb = None
          diffusion = GaussianDiffusion_sam(
                denoise_fn=denoise_fn,
                rrdb_net=rrdb,
                timesteps=hparams['timesteps'],
                loss_type=hparams['loss_type'],
                sam_config=hparams['sam_config']
        )
          
          
         
         
         
         
        
    # optimisation
          params = list(diffusion.named_parameters())
          if not hparams['fix_rrdb']:
              params = [p for p in params if 'rrdb' not in p[0]]
          params = [p[1] for p in params]
          optimizer = torch.optim.Adam(params, lr=0.0001)
          scheduler = build_scheduler(optimizer)
          
          training_step = load_checkpoint(diffusion, optimizer, "/home/cli348/New/checkpoints/Microtubules/", 2)
  

    '''load pretrained model'''

    args.path_helper = set_log_dir('/home/cli348/New/logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
  

    '''segmentation data'''
    transform_train = transforms.Compose([
     transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
       
    ])
    
    transform_gt = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    transform_sr = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
    ])
    


    transform_test = transforms.Compose([
    transforms.ToTensor(),
        transforms.Resize((512, 512)),
        
    ])
 
    
    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, args.data_path, args.target_dir,transform=transform_train, mode='train')
        refuge_test_dataset = REFUGE(args, args.data_path, args.data_path, args.target_dir, transform=transform_test, mode='test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
     
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''

    if args.dataset == 'SR':
       # refuge_train_dataset, refuge_test_dataset = load_super_resolution_dataset(args.data_path, args.data_path, args.target_dir,batch_size=args.b, transform=transform_train, transform_msk=transform_train)
        
      #  refuge_train_dataset = SuperResolutionDataset(args, args.data_path, args.data_path, args.target_dir, transform = transform_train, transform_gt = transform_gt, mode = 'train')
       # refuge_test_dataset = SuperResolutionDataset(args, args.data_path, args.data_path, args.target_dir, transform = transform_test, transform_gt = transform_gt,  mode = 'test')
        refuge_train_dataset = SR(args, transform=transform_train, name='ER', train=True, rootdatapath=args.data_path)
        refuge_test_dataset = SR(args, transform=transform_test, name='ER', train=False, rootdatapath=args.data_path)
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
     
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
      #  refuge_train_dataset = REFUGE(args, args.data_path, transform=transform_train, mode='train')
      #  refuge_test_dataset = REFUGE(args, args.data_path,  transform=transform_test, mode='test')
      


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
 

    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0
    best_PSNR = 0.0
    
    def log_metrics( metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

  
    net.eval()
    diffusion.eval()
    if args.SR==True:
                metrics = function.validation_diffusion(args, nice_test_loader, 0, net, writer, diffusion)
                logger.info(f' results: {metrics} || @ epoch {epoch}.')
                if metrics['psnr'] > best_PSNR:
                    best_PSNR = metrics['psnr']
                    #torch.save({'model': diffusion.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
                    save_checkpoint(diffusion, optimizer, '/home/cli348/New/checkpoints/Devonv', epoch)


    writer.close()


    
                 
def main():
    
    # use bfloat16 for the entire work
    
   
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
 
    
    args = cfg.parse_args()
    print(hparams)
    #set_hparams()
    #dist.init_process_group(backend="nccl")
   # torch.cuda.set_device(args.local_rank)
    #device = torch.device("cuda", args.local_rank)
    #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    GPUdevice = torch.device('cuda', args.gpu_device)
 
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    

    if args.SR == True:
    
       
          hidden_size = hparams['hidden_size']
          dim_mults = hparams['unet_dim_mults']
          dim_mults = [int(x) for x in dim_mults.split('|')]
        
          denoise_fn = Unet(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
          if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
          else:
              rrdb = None
          diffusion = GaussianDiffusion_sam(
                denoise_fn=denoise_fn,
                rrdb_net=rrdb,
                timesteps=hparams['timesteps'],
                loss_type=hparams['loss_type'],
                sam_config=hparams['sam_config']
        )
          
          
         
          diffusion.to("cuda")
         
        
    # optimisation
          params = list(diffusion.named_parameters())
          if not hparams['fix_rrdb']:
              params = [p for p in params if 'rrdb' not in p[0]]
          params = [p[1] for p in params]
          optimizer = torch.optim.Adam(params, lr=0.0001)
          scheduler = build_scheduler(optimizer)
    else: 
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

    '''load pretrained model'''

    args.path_helper = set_log_dir('/home/cli348/New/logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
  

    '''segmentation data'''
    transform_train = transforms.Compose([
     transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
       
    ])
    
    transform_gt = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    transform_sr = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
    ])
    


    transform_test = transforms.Compose([
    transforms.ToTensor(),
        transforms.Resize((512, 512)),
        
    ])
 
    
    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, args.data_path, args.target_dir,transform=transform_train, mode='train')
        refuge_test_dataset = REFUGE(args, args.data_path, args.data_path, args.target_dir, transform=transform_test, mode='test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
     
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''

    if args.dataset == 'SR':
       # refuge_train_dataset, refuge_test_dataset = load_super_resolution_dataset(args.data_path, args.data_path, args.target_dir,batch_size=args.b, transform=transform_train, transform_msk=transform_train)
        
      #  refuge_train_dataset = SuperResolutionDataset(args, args.data_path, args.data_path, args.target_dir, transform = transform_train, transform_gt = transform_gt, mode = 'train')
       # refuge_test_dataset = SuperResolutionDataset(args, args.data_path, args.data_path, args.target_dir, transform = transform_test, transform_gt = transform_gt,  mode = 'test')
        refuge_train_dataset = SR(args, transform=transform_train, name='Microtubules', train=True, rootdatapath=args.data_path)
        refuge_test_dataset = SR(args, transform=transform_test, name='Microtubules', train=False, rootdatapath=args.data_path)
      #  refuge_train_dataset = REFUGE(args, args.data_path, transform=transform_train, mode='train')
     #   refuge_test_dataset = REFUGE(args, args.data_path,  transform=transform_test, mode='test')
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
     
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
      #  refuge_train_dataset = REFUGE(args, args.data_path, transform=transform_train, mode='train')
      #  refuge_test_dataset = REFUGE(args, args.data_path,  transform=transform_test, mode='test')
      
 

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
 

    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0
    best_PSNR = 0.0
    
    def log_metrics( metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    for epoch in range(settings.EPOCH):

       

        # training
        if args.SR:
          net.eval()
          diffusion.train()
        else:
          net.train()
        time_start = time.time()
        if args.SR == True:
            training_step = 0
            diffusion.train()
            for epoch in range(settings.EPOCH):
                    if epoch % args.val_freq == 0 and epoch != 0 :  
                        net.eval()
                        diffusion.eval()
                        if args.SR==True:
                            metrics = function.validation_diffusion(args, nice_test_loader, epoch, net, writer, diffusion)
                            logger.info(f' results: {metrics} || @ epoch {epoch}.')
                            if metrics['psnr'] > best_PSNR:
                                best_PSNR = metrics['psnr']
                                #torch.save({'model': diffusion.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
                                save_checkpoint(diffusion, optimizer, '/home/cli348/New/checkpoints/w2', epoch)
                                
                                
                    net.eval()
                    diffusion.train()
                    loss = function.train_diffusion(args, net, optimizer, nice_train_loader, epoch, writer, diffusion)
                    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
                    
                    scheduler.step(training_step*len(refuge_train_dataset))
                    
                   
                   
        else:
            loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # validation
        
        if epoch % args.val_freq == 0 and epoch != 0 :  
    
            net.eval()
            diffusion.eval()
            if args.SR==True:
                metrics = function.validation_sam(args, nice_test_loader, epoch, net, writer, diffusion)
                logger.info(f' results: {metrics} || @ epoch {epoch}.')
                if metrics['psnr'] > best_PSNR:
                    best_PSNR = metrics['psnr']
                    #torch.save({'model': diffusion.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
                    save_checkpoint(diffusion, optimizer, '/home/cli348/New/checkpoints/Devonv', epoch)
            else:

                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

                if edice > best_dice:
                    best_dice = edice
                    torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    train = True
    if train:
      main()
    else:
      test()

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cfg
from conf import settings
from func_2d.utils import *
import pandas as pd
from torchvision import transforms
from cfg import hparams
args = cfg.parse_args()

from sam2_train.modeling.utils_sr.matlab_resize import imresize
from sam2_train.modeling.utils_sr.utils import  Measure
    

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32

torch.backends.cudnn.benchmark = True

import torch
import math

from rotary_embedding_torch import RotaryEmbedding



import torch
import torch.nn.functional as F


def calculate_ssim(img_pred, img_gt, window_size=11, max_pixel_value=1.0):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img_pred (torch.Tensor): The generated (predicted) image tensor.
        img_gt (torch.Tensor): The ground truth high-resolution image tensor.
        window_size (int): The size of the Gaussian window.
        max_pixel_value (float): The maximum possible pixel value (default 1.0 for normalized images).

    Returns:
        float: SSIM value.
    """
    C1 = (0.01 * max_pixel_value) ** 2
    C2 = (0.03 * max_pixel_value) ** 2

    mu_pred = F.avg_pool2d(img_pred, window_size, 1, 0)
    mu_gt = F.avg_pool2d(img_gt, window_size, 1, 0)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = F.avg_pool2d(img_pred * img_pred, window_size, 1, 0) - mu_pred_sq
    sigma_gt_sq = F.avg_pool2d(img_gt * img_gt, window_size, 1, 0) - mu_gt_sq
    sigma_pred_gt = F.avg_pool2d(img_pred * img_gt, window_size, 1, 0) - mu_pred_gt

    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / (
                (mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))
    return ssim_map.mean().item()


def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer, diffusion: nn.Module = None):
   
   
    net.train()
  
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    feat_sizes = [(256, 256), (128, 128), (64, 64)]


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            # input image and gt masks
            imgs = pack['image'].to(dtype = mask_type, device = GPUdevice)
            target = pack['target'].to(dtype = mask_type, device = GPUdevice)
            masks = pack['mask'].to(dtype = mask_type, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            # click prompt: unsqueeze to indicate only one click, add more click across this dimension
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device = GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device = GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            '''Train image encoder'''                    
            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            # dimension hint for your future use
            # vision_feats: list: length = 3
            # vision_feats[0]: torch.Size([65536, batch, 32])
            # vision_feats[1]: torch.Size([16384, batch, 64])
            # vision_feats[2]: torch.Size([4096, batch, 256])
            # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
            # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
            # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
            
            

            '''Train memory attention to condition on meomory bank'''         
            B = vision_feats[-1].size(1)  # batch size 
            
            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                
            else:
                for element in memory_bank_list:
                    to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
                    to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
                    to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
 
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64) 
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                
                similarity_scores = F.softmax(similarity_scores, dim=1) 
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )


            feats = [feat.permute(1, 2, 0).reshape(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed


            '''prompt encoder'''         
            with torch.no_grad():
                if (ind%5) == 0:
                    points=(coords_torch, labels_torch) # input shape: ((batch, n, 2), (batch, n))
                    flag = True
                else:
                    points=None
                    flag = False

                se, de = net.sam_prompt_encoder(
                    points=points, #(coords_torch, labels_torch)
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])



            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            

            '''memory encoder'''       
            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
                
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             image_embed[batch].reshape(-1).detach()])
            
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     image_embed[batch].reshape(-1).detach()])

           
          
            loss = lossfunc(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            pbar.update()

               

    return epoch_loss / len(train_loader)


def data_augment( img_hr, img_lr, sam_mask):
        sr_scale = hparams['sr_scale']
        img_hr = Image.fromarray(img_hr)
        
        data_position_aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, interpolation=Image.BICUBIC),
        ])
        
        data_color_aug_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        
        img_hr, sam_mask = data_position_aug_transforms([img_hr, sam_mask])
        img_hr = data_color_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr, sam_mask
        
        
        

def train_diffusion(args, net: nn.Module, optimizer, train_loader, epoch, writer, diffusion: nn.Module = None):
    # eval mode
    net.eval()
    

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0

    total_epoch_loss = 0
    

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
         
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            
  
 
    
            target = pack['target'].to(dtype=mask_type, device=GPUdevice)
         

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            '''test'''

            with torch.no_grad():
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(
                        device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                        torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True))  # image_embed

                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(
                        1)  # Shape [batch_size, 16]

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2),
                                                          memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2),
                                                              memory_stack_ori_new.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).reshape(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

                image_embed = feats[-1]
                high_res_feats = feats[:-1]

           
                flag = False
                points = None
                    
            

                se, de = net.sam_prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
               
                
              #  B, C, H, W = pred.shape

                
              #  rotary_emb = RotaryEmbedding(dim=W)
                
              #  rotary_emb = rotary_emb.to(device)
              #  mask_embed_ori = torch.ones(1, 1, H, W, device=pred.device)
                
              
              #  mask_embed_ori = rotary_emb.rotate_queries_or_keys(mask_embed_ori)
                
                
              #  embedding_value = (mask_embed_ori * pred).mean()
                
              
            #    pred = torch.where(pred.bool(),
            #                            torch.full_like(pred, embedding_value),
            #                            torch.zeros_like(pred))
                pred = torch.zeros((1, 1, 1024, 1024)).to(device)
              
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag)

                maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0],
                                                 image_embed[batch].reshape(-1).detach()])

                else:
                    for batch in range(maskmem_features.size(0)):

                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2,
                                                                        dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores)
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < \
                                current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index)
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0],
                                                         image_embed[batch].reshape(-1).detach()])

            if args.SR:
                to_tensor_norm = transforms.Compose([
                  
                #    transforms.ToTensor(),
               
              
                    transforms.Resize((256, 256)),
                ])
                diffusion.train()
                img = to_tensor_norm(imgs)
              
              #  img = imgs.cpu()
              #  processed = []


        #        for i in range(img.size(0)):
                   #       img_sample = img[i].permute(1, 2, 0).cpu().numpy()  
             #             img_lr = imresize(img_sample, 1 / hparams['sr_scale']) 
                    #      img_lr = to_tensor_norm(img_lr).float().to("cuda")  
                 #         processed.append(img_lr)
                      
              
             #   img_lr = torch.stack(processed, dim=0) 
              
               
                
 
                losses, _, _ = diffusion(target, img, imgs,
                                         sam_mask=pred)
                                         
               

                epoch_loss = sum(losses.values())

                optimizer.zero_grad()

                epoch_loss.backward()

                optimizer.step()
                total_epoch_loss += epoch_loss

                pbar.update()

    return total_epoch_loss / len(train_loader)




def validation_diffusion(args, val_loader, epoch, net: nn.Module, writer, diffusion=None):
    net.eval()
    diffusion.eval()
    
    n_val = len(val_loader)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    measure = Measure()
    metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
    ret = {k: 0 for k in metric_keys}
    ret['n_samples'] = 0
    
    # Define output directories
    gen_dir = f"{args.work_dir}/results_{epoch}/validation"
    os.makedirs(f'{gen_dir}/SR', exist_ok=True)
    os.makedirs(f'{gen_dir}/HR', exist_ok=True)
    os.makedirs(f'{gen_dir}/LR', exist_ok=True)
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            target = pack['target'].to(dtype=torch.float32, device=GPUdevice)
            item_names = pack['image_meta_dict']['filename_or_obj']
            
            if args.SR:
                diffusion.eval() 
                
                to_tensor_norm = transforms.Compose([
                   # transforms.ToTensor(),
                  #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              
                    transforms.Resize((128, 128)),
                ])
                diffusion.train()
                img = to_tensor_norm(imgs)
               # img = imgs.cpu()
              #  processed = []
                
             #   for i in range(img.size(0)):
               #     img_sample = img[i].permute(1, 2, 0).cpu().numpy()
               #     img_lr = imresize(img_sample, 1 / hparams['sr_scale'])
               #     img_lr = to_tensor_norm(img_lr).float().to(GPUdevice)
                #    processed.append(img_lr)
                
             #   img_lr = torch.stack(processed, dim=0)
                img_sr, rrdb_out = diffusion.sample(img, imgs, target.shape)
                
                for b in range(img_sr.shape[0]):
                    s = measure.measure(img_sr[b], target[b], img[b], hparams['sr_scale'])
                    for k in metric_keys:
                        ret[k] += s[k]
                    ret['n_samples'] += 1
                    
                    # Save images
                    img_sr_np = (img_sr[b].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                    img_hr_np = (target[b].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                    img_lr_np = (img[b].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                    
                    item_name = os.path.splitext(item_names[b])[0]
                    Image.fromarray(img_sr_np).save(f"{gen_dir}/SR/{item_name}.png")
                    Image.fromarray(img_hr_np).save(f"{gen_dir}/HR/{item_name}.png")
                    Image.fromarray(img_lr_np).save(f"{gen_dir}/LR/{item_name}.png")
            
            pbar.update()
    
    # Compute and save metrics
    metrics = {k: ret[k] / ret['n_samples'] for k in metric_keys}
    
   
    
    return metrics
    
    

    

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # eval mode
    net.eval()

    n_val = len(val_loader)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            '''test'''
            with torch.no_grad():

                """ image encoder """
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                """ memory condition """
                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(
                        device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                        torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True))  # image_embed

                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(
                        1)  # Shape [batch_size, 16]

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2),
                                                          memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2),
                                                              memory_stack_ori_new.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                """ prompt encoder """
                if (ind % 5) == 0:
                    flag = True
                    points = (coords_torch, labels_torch)

                else:
                    flag = False
                    points = None

                se, de = net.sam_prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                """ memory encoder """
                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag)

                maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

                """ memory bank """
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0],
                                                 image_embed[batch].reshape(-1).detach()])

                else:
                    for batch in range(maskmem_features.size(0)):

                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2,
                                                                        dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores)
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < \
                                current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index)
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0],
                                                         image_embed[batch].reshape(-1).detach()])

                # binary mask and calculate loss, iou, dice
                total_loss += lossfunc(pred, masks)
                pred = (pred > 0.5).float()
                temp = eval_seg(pred, masks, threshold)
                total_eiou += temp[0]
                total_dice += temp[1]

                '''vis images'''
                if ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        img_name = na
                        namecat = namecat + img_name + '+'
                    vis_image(imgs, pred, masks,
                              os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                              reverse=False, points=None)

            pbar.update()

    return total_loss / n_val, tuple([total_eiou / n_val, total_dice / n_val])

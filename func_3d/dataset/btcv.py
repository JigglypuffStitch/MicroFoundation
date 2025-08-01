""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
from func_3d.utils import random_click, generate_bbox

class BTCV(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        self.name_list = os.listdir(os.path.join(data_path, mode, 'images'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation

        if mode == 'train':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        name = self.name_list[index]


        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_path = os.path.join(self.data_path, self.mode, 'masks', name)


        img_3d = tiff.imread(img_path)
        data_seg_3d = tiff.imread(mask_path)

        num_frame = data_seg_3d.shape[0]

        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length


        valid_frames = []
        for i in range(num_frame):
            if np.any(data_seg_3d[i, ...] > 0):
                valid_frames.append(i)


        if len(valid_frames) == 0:
            raise ValueError(f"No valid frames with objects found for {name}")

        if len(valid_frames) < video_length:
            selected_frames = valid_frames
        else:
            selected_frames = np.random.choice(valid_frames, video_length, replace=False)

        img_tensor = torch.zeros(len(selected_frames), 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {} 
        pt_dict = {}
        bbox_dict = {}

        for idx, frame_index in enumerate(selected_frames):
            img_frame = img_3d[frame_index, ...]
            img_frame = Image.fromarray(img_frame).convert('RGB')

            mask = data_seg_3d[frame_index, ...]
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}


            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')

            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

            img_frame = img_frame.resize(newsize)
            img_frame = torch.tensor(np.array(img_frame)).permute(2, 0, 1)

            img_tensor[idx, :, :, :] = img_frame
            mask_dict[idx] = diff_obj_mask_dict

            if self.prompt == 'bbox':
                bbox_dict[idx] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[idx] = diff_obj_pt_dict
                point_label_dict[idx] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        

        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }





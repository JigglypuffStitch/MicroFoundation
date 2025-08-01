""" train and test dataset

author jundewu
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
from func_2d.utils import random_click
from func_2d.csbdeep.utils import normalize, axes_dict, axes_check_and_normalize, backend_channels_last, move_channel_for_backend
from func_2d.csbdeep.io import load_training_data
import glob
import tifffile 

class deconvo(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click'):
        super().__init__()
        self.mode = mode
        self.data_path = os.path.join(data_path, mode)  
      
        self.img_folder = os.path.join(self.data_path, 'input')
        self.mask_folder = os.path.join(self.data_path, 'ground_truth')

       
        image_files = [
            f for f in os.listdir(self.img_folder) 
            if os.path.isfile(os.path.join(self.img_folder, f)) and f.lower().endswith('.tif')
        ]
        mask_files = [
            f for f in os.listdir(self.mask_folder) 
            if os.path.isfile(os.path.join(self.mask_folder, f)) and f.lower().endswith('.tif')
        ]

    
        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)
        common_basenames = sorted(image_basenames.intersection(mask_basenames))

        if len(common_basenames) < len(image_basenames):
            missing = len(image_basenames) - len(common_basenames)
          

       
        self.samples = []
        for base_name in common_basenames:
            img_path = os.path.join(self.img_folder, base_name + '.tif')
            with tifffile.TiffFile(img_path) as tif_:
                n_frames = len(tif_.pages)
   
            for i in range(n_frames):
                self.samples.append((base_name, i))

  
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
    
        return len(self.samples)

    def __getitem__(self, index):
    
        base_name, frame_idx = self.samples[index]

      
        img_path = os.path.join(self.img_folder, base_name + '.tif')
        msk_path = os.path.join(self.mask_folder, base_name + '.tif')


        with tifffile.TiffFile(img_path) as tif_:
            frame_img = tif_.pages[frame_idx].asarray()  # shape: (H, W) or (H, W, C)

     
        with tifffile.TiffFile(msk_path) as tif_:
            frame_mask = tif_.pages[frame_idx].asarray()  # shape: (H, W)

       
        pil_img = Image.fromarray(frame_img).convert('RGB')
        pil_msk = Image.fromarray(frame_mask).convert('RGB')

      
        if self.transform is not None:
          
            state = torch.get_rng_state()
            img_t = self.transform(pil_img)  
            torch.set_rng_state(state)
            target =  self.transform(pil_msk)  
            mask_t = torch.as_tensor((self.transform(pil_msk) != 0).float(), dtype=torch.float32)
      

        if self.prompt == 'click':

            point_label_cup, pt_cup = random_click(mask_t.squeeze(0).numpy(), point_label=1)

          
            mask_ori = (mask_t >= 0.5).float()
            mask_resized = F.interpolate(
                mask_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            mask_resized = (mask_resized >= 0.5).float()
        else:
  
            point_label_cup, pt_cup = None, None
            mask_ori = mask_t
            mask_resized = mask_t

        image_meta_dict = {
            'filename_or_obj': base_name + '.tif',
            'frame_idx': frame_idx
        }

        return {
            'image': img_t,          
            'mask': mask_resized,     
            'p_label': point_label_cup,
            'pt': pt_cup,
            'mask_ori': mask_ori,     
            'image_meta_dict': image_meta_dict,
            'target': target,
        }


class PreTrain(Dataset):
    def __init__(self, args, data_path, transform=None, mode='train', prompt='click'):
        """
        读取预先保存好的 image.npz 和 mask.npz，
        并按索引提供样本。
        """
        # data_path 下应包含 image.npz 和 mask.npz
        self.data_path = data_path
        img_npz = os.path.join(self.data_path, 'image.npz')
        msk_npz = os.path.join(self.data_path, 'mask.npz')

        # 加载所有样本到内存
        imgs = np.load(img_npz)
        msks = np.load(msk_npz)
        self.images = imgs['images']  # 形状 (N, H, W, C)
        self.masks = msks['masks']  # 形状 (N, H, W)

        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # 1. 从 np.array 恢复 PIL.Image
        img_arr = self.images[index]  # H×W×C, uint8
        mask_arr = self.masks[index]  # H×W,   uint8 或 bool

        img = Image.fromarray(img_arr).convert('RGB')
        masks = Image.fromarray(mask_arr).convert('RGB')
        mask = Image.fromarray(mask_arr).convert('L')

        # 2. 统一随机变换
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            masks = self.transform(masks)
            # 对单通道掩码做二值化
            mask_t = torch.as_tensor((self.transform(mask) != 0).float(), dtype=torch.float32)
            torch.set_rng_state(state)
        else:
            mask_t = torch.as_tensor(mask_arr, dtype=torch.float32).unsqueeze(0)

        # 3. 点采样（click prompt）
        if self.prompt == 'click':
            # random_click 接受 numpy 二值 mask
            point_label, pt = random_click(mask_t.squeeze(0).numpy(), point_label=1)

            # 原始插值前的掩码
            selected_mask_ori = (mask_t >= 0.5).float()

            # 下采样 / 上采样到指定大小
            selected_mask = F.interpolate(
                selected_mask_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            selected_mask = (selected_mask >= 0.5).float()
        else:
            point_label, pt = None, None
            selected_mask_ori = mask_t
            selected_mask = F.interpolate(
                mask_t.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0, keepdim=True)

        # 4. 元信息
        image_meta_dict = {'index': index}

        return {
            'image': img,
            'mask': selected_mask,
            'p_label': point_label,
            'pt': pt,
            'mask_ori': selected_mask_ori,
            'image_meta_dict': image_meta_dict,
            'target': masks
        }


class REFUGE(Dataset):
    def __init__(self, args, data_path, transform=None, mode='train', prompt='click'):
        """
        读取预先保存好的 image.npz 和 mask.npz，
        并按索引提供样本。
        """
        # data_path 下应包含 image.npz 和 mask.npz
        self.data_path = data_path
        img_npz = os.path.join(self.data_path, 'image.npz')
        msk_npz = os.path.join(self.data_path, 'mask.npz')

        # 加载所有样本到内存
        imgs = np.load(img_npz)
        msks = np.load(msk_npz)
        self.images = imgs['images']   # 形状 (N, H, W, C)
        self.masks  = msks['masks']    # 形状 (N, H, W)

        self.prompt    = prompt
        self.img_size  = args.image_size
        self.mask_size = args.out_size
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # 1. 从 np.array 恢复 PIL.Image
        img_arr  = self.images[index]           # H×W×C, uint8
        mask_arr = self.masks[index]            # H×W,   uint8 或 bool

        img   = Image.fromarray(img_arr).convert('RGB')
        masks = Image.fromarray(mask_arr).convert('RGB')
        mask  = Image.fromarray(mask_arr).convert('L')

        # 2. 统一随机变换
        if self.transform:
            state = torch.get_rng_state()
            img    = self.transform(img)
            masks  = self.transform(masks)
            # 对单通道掩码做二值化
            mask_t = torch.as_tensor((self.transform(mask) != 0).float(), dtype=torch.float32)
            torch.set_rng_state(state)
        else:
            mask_t = torch.as_tensor(mask_arr, dtype=torch.float32).unsqueeze(0)

        # 3. 点采样（click prompt）
        if self.prompt == 'click':
            # random_click 接受 numpy 二值 mask
            point_label, pt = random_click(mask_t.squeeze(0).numpy(), point_label=1)

            # 原始插值前的掩码
            selected_mask_ori = (mask_t >= 0.5).float()

            # 下采样 / 上采样到指定大小
            selected_mask = F.interpolate(
                selected_mask_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            selected_mask = (selected_mask >= 0.5).float()
        else:
            point_label, pt = None, None
            selected_mask_ori = mask_t
            selected_mask = F.interpolate(
                mask_t.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0, keepdim=True)

        # 4. 元信息
        image_meta_dict = {'index': index}

        return {
            'image':           img,
            'mask':            selected_mask,
            'p_label':         point_label,
            'pt':              pt,
            'mask_ori':        selected_mask_ori,
            'image_meta_dict': image_meta_dict,
            'target':          masks
        }
                
class REFUGE(Dataset):
    def __init__(self, args, data_path, transform=None,  mode='train', prompt='click'):
    #    if mode == 'train':
        
        self.data_path = os.path.join(data_path, mode)
       # else:
        #  self.data_path = data_path

        self.img_folder = os.path.join(self.data_path, 'images')
        self.mask_folder = os.path.join(self.data_path, 'masks')

        image_files = [f for f in os.listdir(self.img_folder) if os.path.isfile(os.path.join(self.img_folder, f))]
        mask_files = [f for f in os.listdir(self.mask_folder) if os.path.isfile(os.path.join(self.mask_folder, f))]

        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)

        common_basenames = sorted(image_basenames.intersection(mask_basenames))

        if len(common_basenames) < len(image_basenames):
            missing = len(image_basenames) - len(common_basenames)
            print(f"Warning: There are {missing} image files without corresponding mask files.")

        self.image_filenames = common_basenames
        self.mask_filenames = common_basenames

        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform
       

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        base_filename = self.image_filenames[index]

        img_filename = self._find_file(self.img_folder, base_filename, ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff','.TIF'])
        if img_filename is None:
            raise FileNotFoundError(f"No file found with any extension for: {base_filename} in the image folder.")

        mask_filename = self._find_file(self.mask_folder, base_filename,  ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff','.TIF'])
        if mask_filename is None:
            raise FileNotFoundError(f"No file found with any extension for: {base_filename} in the mask folder.")

        img_path = os.path.join(self.img_folder, img_filename)
        mask_path = os.path.join(self.mask_folder, mask_filename)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        masks = Image.open(mask_path).convert('RGB')
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            masks = self.transform(masks)
            mask = torch.as_tensor((self.transform(mask) != 0).float(), dtype=torch.float32)
            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup = random_click(np.array(mask.squeeze(0)), point_label=1)

            selected_rater_mask_cup_ori = mask
            selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float()

            selected_rater_mask_cup = F.interpolate(
                selected_rater_mask_cup_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()

        image_meta_dict = {'filename_or_obj': img_filename}
        return {
            'image': img,
            'mask': selected_rater_mask_cup,
            'p_label': point_label_cup,
            'pt': pt_cup,
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict': image_meta_dict,
            'target': masks
        }

    def _find_file(self, folder, base_filename, extensions):
        for ext in extensions:
            candidate = base_filename + ext
            if os.path.exists(os.path.join(folder, candidate)):
                return candidate
        return None



def loadData(traindatapath, axes='SCYX', validation_split=0.05):
    print('Load data npz')
    if validation_split > 0:
        (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
    else:
        (X, Y), _, axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
        X_val, Y_val = 0, 1
    print(X.shape, Y.shape)  # (18468, 128, 128, 1) (18468, 256, 256, 1)
    return X, Y, X_val, Y_val
    
      

class SR(data.Dataset):
    def __init__(self, args,transform=None, percent=1.0, scale=2, name='CCPs', train=True, benchmark=False,  test_only=False,
                 rootdatapath='', length=-1, prompt='click'):
        self.length = length
        #self.patch_size = patch_size
        self.rgb_range = 1
        self.name = name
        self.train = train
        self.test_only = test_only
        self.benchmark = benchmark
        self.dir_data = rootdatapath + 'train/%s/my_training_data.npz' % name
        self.dir_demo = rootdatapath + 'test/%s/LR/' % name
        self.transform = transform
        self.prompt = prompt
        
        self.mask_size = args.out_size

        

        self.input_large = (self.dir_demo != '')
        self.scale = scale
        if train:
            X, Y, X_val, Y_val = self.loadData()
            print('np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())

            list_hr, list_lr = Y, X
            list_hr = list_hr[:int(percent * len(list_hr))]
            list_lr = list_lr[:int(percent * len(list_lr))]
        else:
            self.filenames = glob.glob(self.dir_demo + '*.png')
            list_hr, list_lr, name = self._scan()
            self.name = name
           

        self.images_hr, self.images_lr = list_hr, list_lr

    def loadData(self):
       # patch_size = self.patch_size
        X, Y, X_val, Y_val = loadData(self.dir_data)

        N, height, width, c = X.shape
     
        return X, Y, X_val, Y_val

    def _scan(self):
        list_hr, list_lr,  nm = [], [], []
        for fi in self.filenames:
          
            hr = np.array(Image.open(fi.replace('LR', 'GT')).convert('RGB'))
         
            lr = np.array(Image.open(fi).convert('RGB'))
          
          
            
            
            nm.append(fi[len(self.dir_demo):])
            list_hr.append(hr)
            list_lr.append(lr)
          
        return list_hr, list_lr, nm

    def __getitem__(self, idx):
        datamin, datamax = 0, 100
        idx = self._get_index(idx)
        if self.train:
            lr, hr,  filename = self.images_lr[idx], self.images_hr[idx], ''
        else:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx] , self.name[idx]

        hr = normalize(hr, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
       
        
        
    
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(lr)
           
           
         
            mask = self.transform(hr)
            if img.shape[0] ==1:
              
              img = img.repeat(3, 1, 1) 
              target = mask.repeat(3, 1, 1) 
            else:
              target = mask
            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup = random_click(np.array(mask.squeeze(0)), point_label=1)
            selected_rater_mask_cup_ori = mask
            selected_rater_mask_cup = F.interpolate(
                selected_rater_mask_cup_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            
       
        
        image_meta_dict = {'filename_or_obj': filename} 
        return {
            'image': img,
           
            'mask': selected_rater_mask_cup,
            'p_label': point_label_cup,
            'pt': pt_cup,
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict': image_meta_dict,
            'target': target
        }
        


    def __len__(self):
        print('len(self.images_hr)', len(self.images_hr))
        if self.train:
            if self.length < 0:
                return len(self.images_hr)
            else:
                return self.length
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            if self.length < 0:
                return idx % len(self.images_hr)
            else:
                return idx % self.length + self.length * random.randint(0, len(self.images_hr) // self.length - 1)
        else:
            return idx
            
            
            
            
            
class SuperResolutionDataset(Dataset):
    def __init__(self, args, train_dir, test_dir, target_dir, transform=None, transform_gt =None, transform_msk=None, mode='train', prompt='click'):
      
        if mode == 'train':
            self.data_dir = train_dir
            self.fov_range = range(1, 16) 
        elif mode == 'test':
            self.data_dir = test_dir
            self.fov_range = range(16, 17)  
        
        self.target_dir = target_dir
        self.transform = transform
        self.transform_gt = transform_gt
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size


        self.image_filenames = []
        self.mask_filenames = []
        for fov_num in self.fov_range:
            fov_folder = f"FOV{fov_num}"
            fov_path = os.path.join(self.data_dir, fov_folder)
            if not os.path.isdir(fov_path):
                print(f"Warning: {fov_folder} does not exist in {self.data_dir}.")
                continue
            
            for img_file in os.listdir(fov_path):
                if img_file.endswith('.tif'):
                    self.image_filenames.append(os.path.join(fov_path, img_file))
                    
                  
                    mask_filename = f"W800_P200_6mW_Ax1_FOV_{str(fov_num).zfill(2)}_I_t1_SRRF.tif"
                    mask_path = os.path.join(self.target_dir, mask_filename)
                    
                    if os.path.exists(mask_path):
                        self.mask_filenames.append(mask_path)
                    else:
                        print(f"Warning: No mask found for {img_file} in {fov_folder}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_path = self.image_filenames[index]
        mask_path = self.mask_filenames[index]

        img = Image.open(img_path)
   
        img = img.convert('RGB')
   
        mask = Image.open(mask_path).convert('L')

        masks = Image.open(mask_path).convert('RGB')
       
        
        
        

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = self.transform_gt(mask)
            masks = self.transform_gt(masks)
            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup = random_click(np.array(mask.squeeze(0)), point_label=1)
            selected_rater_mask_cup_ori = mask
            selected_rater_mask_cup = F.interpolate(
                selected_rater_mask_cup_ori.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            ).mean(dim=0)
            
       
        print(selected_rater_mask_cup.shape)
        image_meta_dict = {'filename_or_obj': os.path.basename(img_path)} 
        return {
            'image': img,
            'mask': selected_rater_mask_cup,
            'p_label': point_label_cup,
            'pt': pt_cup,
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict': image_meta_dict,
            'target': masks
        }
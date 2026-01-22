#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob 
import os 
# latent_path = "/data1/syliu/ixi_mcx_2025/ixi_mcx_2025_lowres"
# latents = glob.glob(os.path.join(latent_path,"*full.pt"))
# paired_latents = [{"image":x,"image_source":x.replace("full.pt","simple.pt")} for x in latents]
# len(paired_latents)


# In[ ]:


# !pip install rectified_flow_pytorch
# !git clone https://gitee.com/liushiyumath/shiyu_utils.git 


# In[ ]:


from rectified_flow_pytorch.rectified_flow import *
import os

# 验证CUDA_HOME设置
print(f"CUDA_HOME环境变量: {os.environ.get('CUDA_HOME', '未设置')}")
if os.path.exists(os.environ.get('CUDA_HOME', '')):
    print("CUDA_HOME路径存在")
    os.system('ls $CUDA_HOME')
else:
    print("警告: CUDA_HOME路径不存在!")

# 验证CUDA是否可用
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 验证CUDA_VISIBLE_DEVICES设置
    print(f"CUDA_VISIBLE_DEVICES环境变量: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")


# In[ ]:


import monai.data as md 
from tqdm.notebook import tqdm 

import torch 
import monai

from torch.utils.data import Dataset
import glob 
import os 
import torch 
scale = 1 
def load_func(x):
    x = torch.load(x,weights_only=False)[0]
    x = x.float()/scale
    x = x *0.25
    return x
import monai.data as md  

class CT_VESSEL_AVN(Dataset):
    def __init__(self,latent_dir):
        self.latent_dir = latent_dir
        self.files = sorted(glob.glob(os.path.join(latent_dir,"*cropped_A.pt")))
        self.image_dicts=[{
            "image":file,
            "source_image":file.replace("cropped_A","cropped_N")
        } for file in self.files]
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self,idx):
        image_dict = self.image_dicts[idx]
        return {
            "image":load_func(image_dict["image"]),
            "source_image":load_func(image_dict["source_image"]),
        }

class IXI_MCX_paired(Dataset):
    def __init__(self,latent_dir):
        self.latent_dir = latent_dir
        self.files = sorted(glob.glob(os.path.join(latent_dir,"*full.pt")))
        self.image_dicts=[{
            "image":file,
            "source_image":file.replace("full.pt","simple.pt")
        } for file in self.files]
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self,idx):
        image_dict = self.image_dicts[idx]
        return {
            "image":load_func(image_dict["image"]),
            "source_image":load_func(image_dict["source_image"]),
        }

# latent_dir="/mnt/users/notebooks/1/2/maisi_pretrain_finetune/int8latent"
# dataset =CT_VESSEL_AVN(latent_dir)
latent_dir = 'ixi_mcx_2025_latent'
dataset = IXI_MCX_paired(latent_dir)
data_one = dataset[0]
for k,v in data_one.items(): print(k,v.shape)


# In[ ]:


from monai.bundle import ConfigParser
config = ConfigParser()
config.read_config('shiyu_utils/config_maisi3d-rflow.json')
config["autoencoder_def"]["num_splits"]=1
autoencoder =config.get_parsed_content('autoencoder_def',instanitiate=True)
autoencoder.load_state_dict(torch.load("models/autoencoder_epoch273.pt"))
autoencoder.eval()


def test_latent_and_ae():
    latents = sorted(glob.glob(os.path.join(latent_dir,"*.pt")))
    latent = latents[0]
    latent = torch.load(latent,weights_only=False)
    latent = latent.float()
    latent = latent
    latent  = latent / scale 
    latent = latent.float()
    with torch.no_grad(),torch.cuda.amp.autocast(True):
        autoencoder.cuda()
        recon = autoencoder.decode_stage_2_outputs(latent.cuda())
    from shiyu_utils.plot_3d_data import plot_3d_data
    plot_3d_data(recon)
    print(latent.shape,latent.dtype)    
    # =================== 
    latents = sorted(glob.glob(os.path.join(latent_dir,"*.pt")))
    latent = latents[0]
    latent = torch.load(latent,weights_only=False)
    latent = (latent/32).to(torch.float)
    with torch.no_grad(),torch.cuda.amp.autocast(True):
        autoencoder.cuda()
        recon = autoencoder.decode_stage_2_outputs(latent.cuda())
    from shiyu_utils.plot_3d_data import plot_3d_data
    plot_3d_data(recon)
    print(latent.shape,latent.dtype)
    # ==================== 

#test_latent_and_ae()


# In[5]:



from errno import ESTALE
from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer
from monai.networks.nets import DiffusionModelUNet
from torch import nn, pi, cat, stack, from_numpy
from einops import rearrange
class monai_wrapper(torch.nn.Module):
    def __init__(self,model,mean_variance_net=False):
        super().__init__()
        self.model = model
        self.mean_variance_net = mean_variance_net

    def forward(self,x,times,cond=None):
        context = cond 
        timesteps = ((1.- times) * 1000 ).long()
        #for v in[x,context,timesteps]:print(v.shape) if hasattr(v,"shape") else print("v has no shape")
        out= self.model(x,context=context,timesteps=timesteps)
        if self.mean_variance_net:
            mean, log_var = rearrange(out, 'b (c mean_log_var) h w d -> mean_log_var b c h w d', mean_log_var = 2)
            variance = log_var.exp() # variance needs to be positive
            return stack((mean,variance))
        else: 
            return out 


# In[6]:


from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

monai_model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    num_res_blocks=(2,2,2),
    channels=(64,64,128),
    attention_levels=(False,False,True),
    norm_num_groups=32,
    num_head_channels=64,
    cross_attention_dim=None,
    with_conditioning=False,
    use_flash_attention=True,
    )

model = monai_wrapper(monai_model,mean_variance_net=False)
rectified_flow = RectifiedFlow(model,mean_variance_net=False,data_shape=(4,64,64,64),immiscible=False)


# In[ ]:


from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

from tqdm import tqdm 
from rectified_flow_pytorch import Trainer
class MyTrainer(Trainer):
    def _save_2d_in_png(self,sampled,fname):
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def _save_3d_in_png(self,data,fname):
        _,_,h,w,d = data.shape
        hh,ww,dd = h//2,w//2,d//2
        data_xy=data[:,:,hh,:,:]
        data_yz=data[:,:,:,ww,:]
        data_xz=data[:,:,:,:,dd]
        fname_xy=fname.replace(".png","_xy.png")
        fname_yz=fname.replace(".png","_yz.png")
        fname_xz=fname.replace(".png","_xz.png")
        sampled_xy= self._save_2d_in_png(data_xy,fname_xy)
        sampled_yz= self._save_2d_in_png(data_yz,fname_yz)
        sampled_xz= self._save_2d_in_png(data_xz,fname_xz)
        return sampled_xy,sampled_yz,sampled_xz

    def _process_input(self,data):
        noise,cond=None,None
        if isinstance(data,tuple) or isinstance(data,list):
            data,cond=data[0],data[1]
        elif isinstance(data,dict):
            data,noise=data["image"],data["source_image"]
        return data,noise,cond
    
    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)
        dl = cycle(self.dl)
        mock_data = next(dl)
        mock_data,mock_noise,mock_cond =self._process_input(mock_data)
        data_shape = mock_data.shape[1:]
        mock_noise = mock_noise.repeat(self.num_samples//mock_noise.shape[0],1,1,1,1) if hasattr(mock_noise,"shape") else None
        mock_cond = mock_cond.repeat(self.num_samples//mock_cond.shape[0],1,1) if hasattr(mock_cond,"shape") else None
        additional_sample_kwargs = dict()
        if isinstance(eval_model.model, RectifiedFlow):
            additional_sample_kwargs.update(temperature = self.sample_temperature)
            # additional_sample_kwargs.update(noise = mock_noise)
            # additional_sample_kwargs.update(cond = mock_cond)

        with torch.no_grad():
            sampled = eval_model.sample(
                batch_size = self.num_samples,
                data_shape = data_shape,
                noise = mock_noise,
                cond = mock_cond, 
                **additional_sample_kwargs
            )
            global autoencoder
            autoencoder = autoencoder.to(trainer.accelerator.device)
            sampled_collect=[]
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(sampled):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    sampled_collect.append(sample_per_batch)
                sampled = torch.cat(sampled_collect,dim=0)
            #sample_output = self._save_3d_in_png(sampled,fname)

            mock_data_collect=[]
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_data):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    mock_data_collect.append(sample_per_batch)
                mock_data = torch.cat(mock_data_collect,dim=0)
                self._save_3d_in_png(mock_data,fname.replace(".png","_tar.png"))
            mock_noise_collect=[]
            if mock_noise is not None:
                with torch.cuda.amp.autocast(True):
                    for sample_per_batch in tqdm(mock_noise):
                        sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                        mock_noise_collect.append(sample_per_batch)
                    mock_noise = torch.cat(mock_noise_collect,dim=0)
                self._save_3d_in_png(mock_noise,fname.replace(".png","_src.png"))
                return {
                    "input":mock_noise,
                    "target":mock_data,
                    "gen":sampled,
                }
            else:
                return {
                    "gen":sampled,
                }
    def forward(self):

        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            data = next(dl)

            data,noise,cond=self._process_input(data)
            if self.return_loss_breakdown:
                loss, loss_breakdown = self.model(data,noise=noise,cond=cond, return_loss_breakdown = True)
                self.log(loss_breakdown._asdict(), step = step)
            else:
                loss = self.model(data)

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if getattr(self.model, 'use_consistency', False):
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()
            if self.is_main:

                if divisible_by(step, self.save_results_every) or step==1: # we want to sample for the first step to debug sample 

                    sampled = self.sample(fname = str(self.results_folder / f'results.{step}.png'))

                    self.log_images(sampled, step = step)

                if divisible_by(step, self.checkpoint_every) or step==1:
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')
trainer = MyTrainer(
    rectified_flow,
    dataset =[0] *4 , # dataset should be given , here we just use 0 to placeholder
    batch_size=4,
    num_samples=4,
    num_train_steps = 70_000,
    sample_temperature = 1.5,
    checkpoint_every=10000,
    checkpoints_folder="./checkpoints_latent_simple2full",
    results_folder = './results_latent_simple2full',  # samples will be saved periodically to this folder
)

#trainer()


# In[ ]:


def sample_from_trainer(self, input,output,pred):
    eval_model = default(self.ema_model, self.model)
    mock_data,mock_noise,mock_cond = output,input,None
    
    data_shape = mock_data.shape[1:]
    # mock_noise = mock_noise.repeat(self.num_samples//mock_noise.shape[0],1,1,1,1) if hasattr(mock_noise,"shape") else None
    # mock_cond = mock_cond.repeat(self.num_samples//mock_cond.shape[0],1,1) if hasattr(mock_cond,"shape") else None
    mock_data = mock_data.to(self.accelerator.device) if mock_data is not None else None
    mock_noise = mock_noise.to(self.accelerator.device) if mock_noise is not None else None
    mock_cond = mock_cond.to(self.accelerator.device) if mock_cond is not None else None

    additional_sample_kwargs = dict()
    if isinstance(eval_model.model, RectifiedFlow):
        additional_sample_kwargs.update(temperature = self.sample_temperature)

    with torch.no_grad():
        sampled = eval_model.sample(
            batch_size = self.num_samples,
            # data_shape = data_shape,
            noise = mock_noise,
            cond = mock_cond,
            **additional_sample_kwargs
        )
        global autoencoder
        autoencoder = autoencoder.to(self.accelerator.device)

        output_dict={}
        with torch.cuda.amp.autocast(True):
            sampled_collect=[]
            for sample_per_batch in tqdm(sampled):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                sampled_collect.append(sample_per_batch)
            sampled = torch.cat(sampled_collect,dim=0)
            output_dict["gen"]=sampled
        if mock_data is not None:
            mock_data_collect=[]
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_data):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    mock_data_collect.append(sample_per_batch)
                mock_data = torch.cat(mock_data_collect,dim=0)
            output_dict["target"]=mock_data
        if mock_noise is not None:
            mock_noise_collect=[]
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_noise):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    mock_noise_collect.append(sample_per_batch)
                mock_noise = torch.cat(mock_noise_collect,dim=0)
            output_dict["input"]=mock_noise
        return output_dict 


def return_metric_value(gen,target):
    # gen = post_process(gen)
    # target = post_process(target)
    gen = gen.clamp(0,1)
    target = target.clamp(0,1)

    psnr_metric(gen,target)
    ssim_metric(gen,target)
    mae_metric(gen,target)
    mse_metric(gen,target)
    return {
        "psnr": psnr_metric.aggregate(), 
        "ssim": ssim_metric.aggregate(), 
        "mae": mae_metric.aggregate(), 
        "mse": mse_metric.aggregate()
    }

def post_process(gen):
    gen = gen.clamp(0,1)
    gen = gen*10  # 0,1 =>0,10 
    # forward = gen =log(gen+1)
    # backward = gen=gen.exp()-1
    gen = gen.exp()-1
    return gen 

#trainer()
trainer.load('checkpoints_latent_simple2full/checkpoint.70000.pt')
data = next(cycle(trainer.dl))
from monai.metrics import PSNRMetric, SSIMMetric,MAEMetric, MSEMetric
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(3)
mse_metric = MSEMetric()
mae_metric = MAEMetric()
from shiyu_utils.plot_3d_data_temp import (
    plot_mip,
    plot_mip_with_overlay,
    plot_3d_data, 
    plot_3d_data_with_overlay,
    )

import numpy as np 


# In[ ]:


# # step 1 : process data from somewhere , a loader, a ground truth path 
# # pre-process from .nii space to latent .pt space 
# input,output,_ = trainer._process_input(data)
# #output = trainer.sample(fname="temp.png")

# # step 2 : get generation result in .nii space  
# output_dict  = sample_from_trainer(trainer,input,output,None)

# # step 3 : get evaluation in SSIM,PSNR,MAE,MSE,mip visualization and others so on. 
# # optional : use seg . 

# assert "input" in output_dict.keys()
# if "gen" in output_dict.keys() and "target" in output_dict.keys():
#     metric_value = return_metric_value(output_dict["gen"],output_dict["target"])
#     for k,v in metric_value.items():
#         print(k,v)

# diff = output_dict["gen"]-output_dict["target"]
# max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
# min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
# print("max_diff(org range in [0,1]):",max_diff)
# print("min_diff(org range in [0,1]):",min_diff)

# # For generated/target images
# plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
# plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
# plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# # For the absolute difference
# plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[ ]:


# trainer.eval()
# autoencoder.eval()
# all_latents = glob.glob(os.path.join("ixi_mcx_2025_latent","*.pt"))
# import random 
# one_latent = random.choice(all_latents)
# one_paired_latent = {
#     "image":one_latent.replace("_simple.pt","_full.pt"),
#     "source_image":one_latent.replace("_full.pt","_simple.pt")
# }
# one_paired_latent
# one_paired_latent_tensor = {
#     "image":torch.load(one_paired_latent["image"],map_location="cpu",weights_only=False).float()*0.25,
#     "source_image":torch.load(one_paired_latent["source_image"],map_location="cpu",weights_only=False).float()*0.25
# }
# for k,v in one_paired_latent_tensor.items():
#     print(k,v.shape,v.dtype)

# for k,v in next(iter(trainer.dl)).items():
#     print(k,v.shape,v.dtype)

# autoencoder = autoencoder.eval().cuda()


# # step 1 : process data from somewhere , a loader, a ground truth path 
# # pre-process from .nii space to latent .pt space 
# from copy import deepcopy
# data = deepcopy(one_paired_latent_tensor)
# input,output,_ = trainer._process_input(data)
# #output = trainer.sample(fname="temp.png")

# # step 2 : get generation result in .nii space  
# output_dict  = sample_from_trainer(trainer,input,output,None)

# # step 3 : get evaluation in SSIM,PSNR,MAE,MSE,mip visualization and others so on. 
# # optional : use seg . 

# assert "input" in output_dict.keys()
# if "gen" in output_dict.keys() and "target" in output_dict.keys():
#     metric_value = return_metric_value(output_dict["gen"],output_dict["target"])
#     for k,v in metric_value.items():
#         print(k,v)

# diff = output_dict["gen"]-output_dict["target"]
# max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
# min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
# print("max_diff(org range in [0,1]):",max_diff)
# print("min_diff(org range in [0,1]):",min_diff)

# # For generated/target images
# plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
# plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
# plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# # For the absolute difference
# plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[ ]:


import json
json_path = "file_dicts.json"
with open(json_path, "r") as f:
    file_dicts = json.load(f)
import random 
file_dict = random.choice(file_dicts)
from shiyu_utils.maisi_transforms import VAE_Transform
transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    val_patch_size=(160,256,256),
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image","source_image"]
)
transform = transform.transform_dict["ct"]
transform.transforms[-3].scaler.a_min=-10
transform.transforms[-3].scaler.a_max=0 
one_real_data = transform(file_dict)
import monai 
from monai.bundle.config_parser import ConfigParser
import torch 
# !mkdir models
# !cp  /data1/syliu/miccai24_maisi_SR/models/autoencoder_epoch273.pt models/autoencoder_epoch273.pt 
config= ConfigParser()
config.read_config("shiyu_utils/config_maisi3d-rflow.json")
autoencoder = config.get_parsed_content("autoencoder_def")
autoencoder.load_state_dict(torch.load("models/autoencoder_epoch273.pt"))
autoencoder=autoencoder.eval().cuda()
for k,v in one_real_data.items():
    print(k,v.shape)
with torch.no_grad(),torch.cuda.amp.autocast(True):
    one_real_latent ={}
    for k,v in one_real_data.items():
        one_real_latent[k] = autoencoder.encode_stage_2_inputs(v[None,...].cuda().float())
        one_real_latent[k] *= 0.25
    for k,v in one_real_latent.items():
        print(k,v.shape)


# step 1 : process data from somewhere , a loader, a ground truth path 
# pre-process from .nii space to latent .pt space 
from copy import deepcopy
data = deepcopy(one_real_latent)
input,output,_ = trainer._process_input(data)
#output = trainer.sample(fname="temp.png")

# step 2 : get generation result in .nii space  
output_dict  = sample_from_trainer(trainer,input,output,None)

# step 3 : get evaluation in SSIM,PSNR,MAE,MSE,mip visualization and others so on. 
# optional : use seg . 
output_dict["target"]=one_real_data["source_image"][None,...].cuda().float()
assert "input" in output_dict.keys()
if "gen" in output_dict.keys() and "target" in output_dict.keys():
    metric_value = return_metric_value(output_dict["gen"],output_dict["target"])
    for k,v in metric_value.items():
        print(k,v)

diff = output_dict["gen"]-output_dict["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)

output_dict["gen"]=(10*output_dict["gen"]).exp()-1 
output_dict["target"]=(10*output_dict["target"]).exp()-1
output_dict["input"]=(10*output_dict["input"]).exp()-1
diff = output_dict["gen"]-output_dict["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[ ]:


# import json
# json_path = "file_dicts.json"
# with open(json_path, "r") as f:
#     file_dicts = json.load(f)
# import random 
# file_dict = random.choice(file_dicts)
file_dict={
    "image":"results_run_full/MCX_results_log.nii.gz",
    "source_image":"results_run_simple/MCX_results_log.nii.gz",
}
from shiyu_utils.maisi_transforms import VAE_Transform
transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    val_patch_size=(160,256,256),
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image","source_image"]
)
transform = transform.transform_dict["ct"]
transform.transforms[-3].scaler.a_min=-10
transform.transforms[-3].scaler.a_max=0 
one_real_data = transform(file_dict)
import monai 
from monai.bundle.config_parser import ConfigParser
import torch 
# !mkdir models
# !cp  /data1/syliu/miccai24_maisi_SR/models/autoencoder_epoch273.pt models/autoencoder_epoch273.pt 
config= ConfigParser()
config.read_config("shiyu_utils/config_maisi3d-rflow.json")
autoencoder = config.get_parsed_content("autoencoder_def")
autoencoder.load_state_dict(torch.load("models/autoencoder_epoch273.pt"))
autoencoder=autoencoder.eval().cuda()
for k,v in one_real_data.items():
    print(k,v.shape)
with torch.no_grad(),torch.cuda.amp.autocast(True):
    one_real_latent ={}
    for k,v in one_real_data.items():
        one_real_latent[k] = autoencoder.encode_stage_2_inputs(v[None,...].cuda().float())
        one_real_latent[k] *= 0.25
    for k,v in one_real_latent.items():
        print(k,v.shape)


# step 1 : process data from somewhere , a loader, a ground truth path 
# pre-process from .nii space to latent .pt space 
from copy import deepcopy
data = deepcopy(one_real_latent)
input,output,_ = trainer._process_input(data)
#output = trainer.sample(fname="temp.png")

# step 2 : get generation result in .nii space  
output_dict  = sample_from_trainer(trainer,input,output,None)

# step 3 : get evaluation in SSIM,PSNR,MAE,MSE,mip visualization and others so on. 
# optional : use seg . 
output_dict["target"]=one_real_data["source_image"][None,...].cuda().float()
assert "input" in output_dict.keys()
if "gen" in output_dict.keys() and "target" in output_dict.keys():
    metric_value = return_metric_value(output_dict["gen"],output_dict["target"])
    for k,v in metric_value.items():
        print(k,v)

diff = output_dict["gen"]-output_dict["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)

output_dict["gen"]=(10*output_dict["gen"]).exp()-1 
output_dict["target"]=(10*output_dict["target"]).exp()-1
output_dict["input"]=(10*output_dict["input"]).exp()-1
diff = output_dict["gen"]-output_dict["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output_dict["gen"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["target"], cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output_dict["input"], cmap_main="magma", vmin_main=0, vmax_main=1)

# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[ ]:


volume_lists= [
    output_dict["gen"],
    output_dict["target"],
    output_dict["input"],
]
import nibabel as nib 
vol= nib.load("results_run_full/MCX_input_vol.nii.gz").get_fdata()[...,0]
print(vol.shape)

from shiyu_utils.plot_3d_data import plot_3d_data 
for v in volume_lists + [vol]:
    plot_3d_data(v)


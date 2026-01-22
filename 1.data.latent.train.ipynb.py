#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob 
import os 
latent_path = "/data1/syliu/ixi_mcx_2025/ixi_mcx_2025_lowres"
latents = glob.glob(os.path.join(latent_path,"*full.pt"))
paired_latents = [{"image":x,"image_source":x.replace("full.pt","simple.pt")} for x in latents]
len(paired_latents)


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


# In[6]:


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
            sample_output = self._save_3d_in_png(sampled,fname)

            mock_data_collect=[]
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_data):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    mock_data_collect.append(sample_per_batch)
                mock_data = torch.cat(mock_data_collect,dim=0)
            sample_output = self._save_3d_in_png(mock_data,fname.replace(".png","_tar.png"))
            mock_noise_collect=[]
            if mock_noise is not None:
                with torch.cuda.amp.autocast(True):
                    for sample_per_batch in tqdm(mock_noise):
                        sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                        mock_noise_collect.append(sample_per_batch)
                    mock_noise = torch.cat(mock_noise_collect,dim=0)
                mock_noise = self._save_3d_in_png(mock_noise,fname.replace(".png","_src.png"))
                return sample_output,mock_noise
            else:
                return sample_output
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
    dataset = dataset,
    batch_size=4,
    num_samples=4,
    num_train_steps = 70_000,
    sample_temperature = 1.5,
    checkpoint_every=10000,
    checkpoints_folder="./checkpoints_latent_simple2full",
    results_folder = './results_latent_simple2full',  # samples will be saved periodically to this folder
)

trainer()


# In[7]:


1


#!/usr/bin/env python
# coding: utf-8

# In[39]:


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
# print cureent cuda device gpu memory : 
print(f"当前CUDA设备内存: {torch.cuda.memory_allocated()}")


# In[40]:


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
    return torch.load(x,map_location="cpu")[0]/scale
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
latent_dir = 'ixi_mcx_2025_lowres'
dataset = IXI_MCX_paired(latent_dir)
data_one = dataset[0]
for k,v in data_one.items(): print(k,v.shape)


# In[41]:


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
    in_channels=1,
    out_channels=1,
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


# In[42]:


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
            self._save_3d_in_png(sampled,fname)

            self._save_3d_in_png(mock_data,fname.replace(".png","_tar.png"))
            if mock_noise is not None:
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
    dataset = dataset,
    batch_size=4,
    num_samples=4,
    num_train_steps = 300_000,
    sample_temperature = 1.5,
    checkpoint_every=10000,
    checkpoints_folder="./checkpoints_lowres_simple2full",
    results_folder = './results_lowres_simple2full',  # samples will be saved periodically to this folder
)

#trainer()


# In[45]:


#trainer()
trainer.load('checkpoints_lowres_simple2full/checkpoint.170000.pt')
output = trainer.sample(fname="temp.png")

from monai.metrics import PSNRMetric, SSIMMetric,MAEMetric, MSEMetric
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(3)
mse_metric = MSEMetric()
mae_metric = MAEMetric()

def return_metric_value(gen,target):
    #gen = post_process(gen)
    #target = post_process(target)
    gen = gen.clamp(0,1)
    target = target.clamp(0,1)

    psnr_metric(gen,target)
    ssim_metric(gen,target)
    mae_metric(gen,target)
    mse_metric(gen,target)
    return psnr_metric.aggregate(), ssim_metric.aggregate(), mae_metric.aggregate(), mse_metric.aggregate()

def post_process(gen):
    gen = gen.clamp(0,1)
    gen = gen*10  # 0,1 =>0,10 
    # forward = gen =log(gen+1)
    # backward = gen=gen.exp()-1
    gen = gen.exp()-1
    return gen 

metric_value = return_metric_value(output["gen"],output["target"])
print("psnr,ssim,mae,mse:",metric_value)

from shiyu_utils.plot_3d_data_temp import (
    plot_mip,
    plot_mip_with_overlay,
    plot_3d_data, 
    plot_3d_data_with_overlay,
    )

import numpy as np 
diff = output["gen"]-output["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output["gen"].clamp(0,1), cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output["target"].clamp(0,1), cmap_main="magma", vmin_main=0, vmax_main=1)

# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[46]:


#trainer()
trainer.load('checkpoints_lowres_simple2full/checkpoint.170000.pt')
output = trainer.sample(fname="temp.png")

from monai.metrics import PSNRMetric, SSIMMetric,MAEMetric, MSEMetric
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(3)
mse_metric = MSEMetric()
mae_metric = MAEMetric()

def return_metric_value(gen,target):
    #gen = post_process(gen)
    #target = post_process(target)
    gen = gen.clamp(0,1)
    target = target.clamp(0,1)

    psnr_metric(gen,target)
    ssim_metric(gen,target)
    mae_metric(gen,target)
    mse_metric(gen,target)
    return psnr_metric.aggregate(), ssim_metric.aggregate(), mae_metric.aggregate(), mse_metric.aggregate()

def post_process(gen):
    gen = gen.clamp(0,1)
    gen = gen*10  # 0,1 =>0,10 
    # forward = gen =log(gen+1)
    # backward = gen=gen.exp()-1
    gen = gen.exp()-1
    return gen 

metric_value = return_metric_value(output["gen"],output["target"])
print("psnr,ssim,mae,mse:",metric_value)

from shiyu_utils.plot_3d_data_temp import (
    plot_mip,
    plot_mip_with_overlay,
    plot_3d_data, 
    plot_3d_data_with_overlay,
    )

import numpy as np 
diff = output["gen"]-output["target"]
max_diff = np.percentile(diff.float().detach().cpu().numpy(),99)
min_diff = np.percentile(diff.float().detach().cpu().numpy(),1)
print("max_diff(org range in [0,1]):",max_diff)
print("min_diff(org range in [0,1]):",min_diff)

# For generated/target images
plot_mip(output["gen"].clamp(0,1), cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output["target"].clamp(0,1), cmap_main="magma", vmin_main=0, vmax_main=1)
plot_mip(output["input"].clamp(0,1), cmap_main="magma", vmin_main=0, vmax_main=1)
# For the absolute difference
plot_mip(diff.abs(), cmap_main="hot", vmin_main=0, vmax_main=1)


# In[47]:


# Histogram matching: adjust output['gen'] to match histogram of output['target']
import torch
def histogram_match_torch(source, reference, num_bins=1000, mask_source=None, mask_reference=None):
    
    src = source.detach().float().cpu().clone()
    ref = reference.detach().float().cpu()
    
    def flatten_per_channel(t):
        if t.dim()==5:
            b,c= t.shape[:2]
            return t.reshape(b*c, -1)
        elif t.dim()==4:
            c = t.shape[0]
            return t.reshape(c, -1)
        else:
            return t.reshape(1, -1)
    src_flat = flatten_per_channel(src)
    ref_flat = flatten_per_channel(ref)
    
    if mask_source is not None:
        m = flatten_per_channel((mask_source>0).float())
        src_flat = torch.stack([a[m_i.bool()] if m_i.any() else a for a,m_i in zip(src_flat, m)])
    if mask_reference is not None:
        m = flatten_per_channel((mask_reference>0).float())
        ref_flat = torch.stack([a[m_i.bool()] if m_i.any() else a for a,m_i in zip(ref_flat, m)])
    
    src_flat = src_flat.clamp(0,1)
    ref_flat = ref_flat.clamp(0,1)
    eps = 1e-7
    bin_edges = torch.linspace(0., 1., steps=num_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    matched = []
    for i in range(src_flat.shape[0]):
        s = src_flat[i]
        r = ref_flat[min(i, ref_flat.shape[0]-1)]
        s_hist = torch.histc(s, bins=num_bins, min=0.0, max=1.0) + eps
        r_hist = torch.histc(r, bins=num_bins, min=0.0, max=1.0) + eps
        s_cdf = torch.cumsum(s_hist, dim=0); s_cdf = s_cdf / s_cdf[-1]
        r_cdf = torch.cumsum(r_hist, dim=0); r_cdf = r_cdf / r_cdf[-1]
        def interp_monotonic(x, xp, fp):
            x = x.clamp(0,1)
            xp = xp.clamp(0,1)
            idx = torch.searchsorted(xp, x, right=False)
            idx = idx.clamp(1, xp.numel()-1)
            x0 = xp[idx-1]; x1 = xp[idx]
            y0 = fp[idx-1]; y1 = fp[idx]
            t = (x - x0) / (x1 - x0 + eps)
            return y0 + t * (y1 - y0)
        mapping = interp_monotonic(s_cdf, r_cdf, bin_centers)
        s_bins = torch.clamp((s * (num_bins-1)).long(), 0, num_bins-1)
        s_mapped = mapping[s_bins]
        matched.append(s_mapped)
    matched = torch.stack(matched, dim=0)
    if source.dim()==5:
        b,c = source.shape[:2]
        out = matched.reshape(b, c, *source.shape[2:])
    elif source.dim()==4:
        c = source.shape[0]
        out = matched.reshape(c, *source.shape[1:])
    else:
        out = matched.reshape(*source.shape)
    return out

def hist_match_loop(source, reference, num_bins=1000, mask_source=None, mask_reference=None,iter=10,):
    for i in range(iter):
        gen_matched = histogram_match_torch(source, reference, num_bins=num_bins, mask_source=mask_source, mask_reference=mask_reference)
        gen_matched = gen_matched.clamp(0,1)
        source = gen_matched.to(source.device)
        print('Histogram matching applied. New gen range:', float(gen_matched.min()), float(gen_matched.max()))
    return gen_matched


# Apply histogram matching to output['gen'] against output['target']
gen_matched = histogram_match_torch(output['gen'], output['input'], num_bins=1000)
#gen_matched = hist_match_loop(output['gen'], output['input'], num_bins=1000,iter=5)
gen_matched = gen_matched.clamp(0,1)
output['gen'] = gen_matched.to(output["target"].device)
print('Histogram matching applied. New gen range:', float(gen_matched.min()), float(gen_matched.max()))

# Recompute metrics and visualize
metric_value = return_metric_value(output['gen'], output['target'])
print('After hist-match psnr, ssim, mae, mse:', metric_value)
plot_mip(output['gen'].clamp(0,1), cmap_main='magma', vmin_main=0, vmax_main=1,title="gen_mip")
plot_mip(output['target'].clamp(0,1), cmap_main='magma', vmin_main=0, vmax_main=1,title="target_mip")
plot_mip(output['input'].clamp(0,1), cmap_main='magma', vmin_main=0, vmax_main=1,title="input_mip")
diff = (output['gen'] - output['target']).abs()
plot_mip(diff, cmap_main='hot', vmin_main=0, vmax_main=1)


from tqdm import tqdm 
from monai.bundle import ConfigParser
import argparse
import json 
import monai 
from monai.networks.nets.autoencoderkl import Encoder
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import RFlowScheduler
import torch 
import os 
import torch.nn as nn 



def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)
def build_args_from_path(json_path):
    args = argparse.Namespace()
    with open(json_path, "r") as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        setattr(args, k, v)
    return args
THIS_FILE_PATH_FOLDER=os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHT_PATH={
    "encoder":os.path.join(THIS_FILE_PATH_FOLDER,"model_encoder.pth"),
    "diff_unet":os.path.join(THIS_FILE_PATH_FOLDER,"model_diff_unet.pth"),
}

@torch.no_grad()
def encode(x,encoder):
    return encoder(x)

@torch.no_grad()
def decode(z,decoder,noise_scheduler,upscaler):
    z = torch.nn.functional.layer_norm(z,z.shape[1:])
    z = upscaler(z)
    noise_scheduler.set_timesteps(num_inference_steps=25,input_img_size_numel=z.shape[1:])
    timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat([timesteps[1:],torch.tensor([0])],dim=0)
    b,_,h,w,d = z.shape
    xt = torch.randn(b,1,h,w,d).to(z.device)
    for t, next_t in tqdm(zip(timesteps,all_next_timesteps),total=len(timesteps)):
        vt = decoder(torch.cat([xt,z],dim=1),torch.tensor([t]).long().to(xt.device))
        xt,_ = noise_scheduler.step(vt,t,xt,next_t)
    return xt

@torch.no_grad()
def recon_image(x,encoder,decoder,noise_scheduler,upscaler):
    z = encode(x,encoder)
    
    xt = decode(z,decoder,noise_scheduler,upscaler)
    return xt

def recon_image_fn(encoder,decoder,noise_scheduler,upscaler):
    def _fn(x):
        return recon_image(x,encoder,decoder,noise_scheduler,upscaler)
    return _fn

def get_recon_fn(args):
    device=torch.device(args.device)
    encoder,diff_unet,noise_scheduler,upscaler = define_instance(args,"encoder"),define_instance(args,"diff_unet"),define_instance(args,"noise_scheduler"),define_instance(args,"upscaler")
    encoder,diff_unet= encoder.to(device),diff_unet.to(device)
    encoder.load_state_dict(torch.load(MODEL_WEIGHT_PATH["encoder"]))
    diff_unet.load_state_dict(torch.load(MODEL_WEIGHT_PATH["diff_unet"]))
    return recon_image_fn(encoder,diff_unet,noise_scheduler,upscaler)

class DiffusionDecoder(nn.Module):
    def __init__(self,decoder,noise_scheduler,upscaler):
        super().__init__()
        self.decoder = decoder
        self.noise_scheduler = noise_scheduler
        self.upscaler = upscaler
    def __call__(self,z):
        return decode(z,self.decoder,self.noise_scheduler,self.upscaler)
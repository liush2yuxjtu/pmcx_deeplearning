#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
json_path = "file_dicts.json"
with open(json_path, "r") as f:
    file_dicts = json.load(f)
file_dicts[0]
import nibabel as nib 
for v in file_dicts[0].values():
    if v.endswith(".nii.gz"):
        img = nib.load(v).get_fdata()
        print(img.shape)
file_dict_one = []
for file_dict in file_dicts:
    file_dict_one.append({"image":file_dict["image"]})
    file_dict_one.append({"image":file_dict["source_image"]})
print(len(file_dict_one))


# In[2]:


import monai 
from monai.bundle.config_parser import ConfigParser
import torch 
# !mkdir models
# !cp  /data1/syliu/miccai24_maisi_SR/models/autoencoder_epoch273.pt models/autoencoder_epoch273.pt 
config= ConfigParser()
config.read_config("shiyu_utils/config_maisi3d-rflow.json")
autoencoder = config.get_parsed_content("autoencoder_def")
autoencoder.load_state_dict(torch.load("models/autoencoder_epoch273.pt"))


# In[3]:


from shiyu_utils.maisi_transforms import VAE_Transform
transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    val_patch_size=(160,256,256),
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image"]
)
transform = transform.transform_dict["ct"]
transform.transforms[-3].scaler.a_min=-10
transform.transforms[-3].scaler.a_max=0
transform


# In[4]:


get_ipython().system('ls {ixi_mcx_2025_latent}')


# In[5]:


import monai.data as md 
dataset = md.Dataset(data=file_dict_one,transform=transform)
dataloader = md.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=8)
save_data_root ="ixi_mcx_2025_latent"
import os 
import torch 
import shutil 
os.makedirs(save_data_root,exist_ok=True)
autoencoder = autoencoder.cuda().eval()
from tqdm import tqdm
for batch in tqdm(dataloader):
    file_name=batch["image"].meta["filename_or_obj"][0].replace(".nii","").replace(".gz","")
    base_name = os.path.basename(file_name)
    label_name = os.path.basename(os.path.dirname(file_name))
    subject_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_name))))
    save_path = os.path.join(save_data_root,subject_name+"_"+base_name+"_"+label_name+".pt")
    if os.path.exists(save_path):
        continue
    with torch.no_grad(),torch.cuda.amp.autocast(True):
        latent = autoencoder.encode_stage_2_inputs(batch['image'].float().cuda())
        latent = latent.cpu().detach()
        torch.save(latent,save_path.replace(".nii.gz",".pt"))
        print("saving latent:",save_path)


# In[6]:


batch = next(iter(dataloader))


# In[7]:


batch["image"].meta["filename_or_obj"][0]


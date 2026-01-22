#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os 
import glob 


# In[25]:


data_root="../IXI_mcx"
subjects = glob.glob(os.path.join(data_root, "*"))

subject = subjects[0]
folders = glob.glob(os.path.join(subject, "pmcx_output","*"))
folder = folders[0]
files = glob.glob(os.path.join(folder, "*"))
files[0]


# In[33]:


os.path.dirname(os.path.dirname(files[0])).replace("pmcx_output","final_tissues.nii.gz")


# In[34]:


files = glob.glob(os.path.join(data_root, "*","pmcx_output","full","*log.nii.gz"),recursive=True)
file_dicts =[{
    "image":file,
    "source_image":file.replace("full","simple"),
    "final_tissues":os.path.dirname(os.path.dirname(files[0])).replace("pmcx_output","final_tissues.nii.gz"),
    #"subject":file.split("/")[-4],
} for file in files]
for d in file_dicts: 
    for v in d.values(): 
        if not os.path.exists(v): print(v)


# In[35]:


# write file_dicts to json
import json
with open("file_dicts.json", "w") as f:
    json.dump(file_dicts, f, indent=2)


# In[39]:


import os 
import glob 
path = "/data1/syliu/ixi_mcx_2025/ixi_mcx_2025_lowres/*"
files = glob.glob(path)
print(len(files))
folder_size = sum(os.path.getsize(f) for f in files)
print(f"{folder_size / (1024 * 1024 * 1024):.2f} GB")
os.path.getsize(files[0])/ (1024 * 1024 * 1024)*20_000


# In[6]:


import json 
json_path  = "file_dicts.json"
json_coutent = json.load(open(json_path, "r"))
json_coutent[:3]
import os 
for subject in json_coutent:
    for k,v in subject.items():
        if not os.path.exists(v):
            print(f"{k}:{v} not exists")


# In[ ]:


# dataset 2 : 
# 10 . 

# file_dicts.json 


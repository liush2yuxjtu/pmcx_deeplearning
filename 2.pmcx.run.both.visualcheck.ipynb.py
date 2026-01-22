#!/usr/bin/env python
# coding: utf-8

# In[22]:


full_folder = "results_run_full"
simple_folder="results_run_simple"
import os 
import glob 


# In[23]:


from shiyu_utils.plot_3d_data_temp import (
    plot_mip,
    plot_mip_with_overlay,
    plot_3d_data, 
    plot_3d_data_with_overlay,
    best_slices_global_peak,
    plot_3d_data_coord,
    plot_3d_data_auto,
    )


# In[32]:


import nibabel as nib 
def fetch_data(path):
    data = nib.load(path)
    data = data.get_fdata()[...,0]
    return data
seg = fetch_data(os.path.join(full_folder,"MCX_input_vol.nii.gz"))
plot_3d_data(seg)
data = fetch_data(os.path.join(full_folder,"MCX_results_log.nii.gz"))
plot_3d_data(data)
z,y,x=best_slices_global_peak(data)
plot_3d_data_coord(data,x=x,y=y,z=z)
plot_mip_with_overlay(
    image= data,
    seg_image=seg,
    alpha=0.5,
    cmap_main="hot",
    cmap_seg="gray",
    vmin_main=-10,
    vmax_main=0
)
plot_3d_data_auto(
    image= data,
    seg_image=seg,
    alpha=0.5,
    cmap_main="hot",
    cmap_seg="gray",
    vmin_main=-10,
    vmax_main=0,
)


# In[33]:


get_ipython().run_line_magic('pinfo', 'plot_3d_data_auto')


# In[36]:


plot_3d_data_auto(
    image= data,
    seg_image=seg,
    alpha=0.5,
    cmap_main="hot",
    cmap_seg="gray",
    vmin_main=-10,
    vmax_main=0,
    strategy="slice_sum",
)


# In[ ]:





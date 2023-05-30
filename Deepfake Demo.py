#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/AliaksandrSiarohin/first-order-model')


# In[2]:


get_ipython().run_line_magic('cd', 'first-order-model')
get_ipython().system('pip install ffmpeg-python')
get_ipython().system('pip install imageio')
get_ipython().system('pip install imageio-ffmpeg')


# In[26]:


get_ipython().system('pip3 install torch torchvision torchaudio')


# In[30]:


import imageio 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize 
from IPython.display import HTML
import warnings 
warnings.filterwarnings("ignore")


# In[31]:


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))
    
    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=l), animated=True)
        plt.axis('off')
        ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
        plt.close()
        return ani


# In[2]:


from demo import load_checkpoints
generator, kp_detector = load_checkpoints (config_path='../first-orfer-model/config/vox-256.yaml', checkpoint_path='../vox-cpk.pth.tar')


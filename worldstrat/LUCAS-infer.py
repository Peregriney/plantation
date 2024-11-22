import numpy as np
import torch
from glob import glob
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, Lambda
import pandas as pd
from pathlib import Path
import os

kws = dict(
    input_size=(250, 250),
    output_size=(780, 780),
    chip_size=(50, 50),
    revisits={"lr": 1, "lrc": 1, "hr": 0, "hr_pan": 0},
    normalize_lr=True,
    sclcolor=False,
    lrc_filter_values=None,
    lrc_filter_thres=0.9,
    root="mountdata/",
   # multiprocessing_manager=multiprocessing_manager,
   # list_of_aois=list_of_aois,
    calculate_median_std=False,
    radiometry_depth=12,
    lr_bands="all",
    num_workers=0,
    max_epochs=-1,
)

import tifffile
from src.lightning_modules import LitModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

tpath = 'mountdata/euroorchards2-50m/Image_000000000000000028ec.tif'
tiff_image = tifffile.imread(tpath)
torch_image = torch.from_numpy(np.array(tiff_image, dtype=np.float32))
torch_image = torch_image.transpose(0,2)
print(torch_image)
print(type(torch_image))
print(torch_image.shape)

S2_SN7_MEAN = torch.Tensor(
    [
        0.0769,
        0.0906,
        0.1162,
        0.1318,
        0.1619,
        0.2275,
        0.2500,
        0.2588,
        0.2667,
        0.2684,
        0.2461,
        0.1874,
    ]
).to(torch.float64)

S2_SN7_STD = torch.Tensor(
    [
        0.1708,
        0.1672,
        0.1614,
        0.1677,
        0.1698,
        0.1513,
        0.1490,
        0.1515,
        0.1461,
        0.2365,
        0.1472,0.1418,]
).to(torch.float64)

#torch_image = torch.nn.functional.normalize(torch_image, p=1, dim=2)
print(torch_image)
normalize_s2 = Normalize(
        mean=S2_SN7_MEAN,std=S2_SN7_STD
        )
print('AFTER SECOND TRAINING_DATA_LEVEL Normalization')
torch_image = normalize_s2(torch_image)
#print(torch_image)
       
torch_image = torch_image[:,1:51,1:51]
torch_image = torch_image.unsqueeze(0).unsqueeze(0)
print(torch_image.shape)

def load_model(checkpoint, device):
    """ Loads a model from a checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint.

    Returns
    -------
    model : lightning_modules.LitModel
        The model.
    """    
    model = LitModel.load_from_checkpoint(checkpoint).eval()
    return model.to(device)

checkpoint_path = f"pretrained_model/model.ckpt"
model = load_model(checkpoint_path, device)
y_hat = model(torch_image)

'''
return make_dataloaders(
    subdir={"lr": "", "lrc": "", "hr": "", "hr_pan": ""},#, "metadata": ""},
    bands_to_read={
            "lr": lr_bands,
            "lrc": None,
            "hr": SPOT_RGB_BANDS,
            "hr_pan": None,
            # "metadata": None,
    },
    transforms=transforms,
    number_of_scenes_per_split={
            "train": kws["train_split"],
            "val": kws["val_split"],
            "test": kws["test_split"],
    },
    file_postfix={
            "lr": "-L2A_data.tiff",
            "lrc": "-CLM.tiff",
            "hr": hr_postfix,
            "hr_pan": hr_pan_postfix,
            # "metadata": hr_postfix,
    },
        **kws,
 )


'''

print(y_hat)
print(y_hat.shape)
from src.plot import showtensor
from torchvision.utils import make_grid
showtensor(
        make_grid(
            torch_image[:, :, [4,3,2]].cpu(),
            nrow=1,
            normalize=True,
            scale_each=True,
        ),
        figsize=10,
    

    )
pred = y_hat.cpu().detach().numpy()

np.save('pred',pred)


S2_SN7_MEAN = torch.Tensor(
    [
        0.0769,
        0.0906,
        0.1162,
        0.1318,
        0.1619,
        0.2275,
        0.2500,
        0.2588,
        0.2667,
        0.2684,
        0.2461,
        0.1874,
    ]
).to(torch.float64)

S2_SN7_STD = torch.Tensor(
    [
        0.1708,
        0.1672,
        0.1614,
        0.1677,
        0.1698,
        0.1513,
        0.1490,
        0.1515,
        0.1461,
        0.2365,
        0.1472,
        0.1418,
    ]
).to(torch.float64)


for filename in os.listdir('mountdata/euroorchards2-50m'):
    if filename.endswith('f.tif'):  # Modify the file extension as needed
        file_path = os.path.join('mountdata/euroorchards2-50m',filename)
        tiff_image = tifffile.imread(file_path)
        torch_image = torch.from_numpy(np.array(tiff_image,dtype=np.float32))
        torch_image = torch_image.transpose(0,2)
        # Load and preprocess the image
          # Add batch dimension
        #normalize_s2 = torchvision.transforms.Normalize(
                #)
        torch_image = torch.nn.functional.normalize(torch_image,p=1,dim=2)
        torch_image = torch_image.unsqueeze(0).unsqueeze(0)
        normalize_s2 = Normalize(
        mean=S2_SN7_MEAN,std=S2_SN7_STD
        )
        torch_image = normalize_s2(torch_image)
        # Make a prediction using the model
        y_hat = model(torch_image)
        pred = y_hat.cpu().detach().numpy()
        
        output_path = os.path.join('mountdata/euroorchardsSR',filename)
        np.save(output_path, pred)
        print('saved')

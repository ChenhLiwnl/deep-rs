import os
import argparse
import shutil
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from models.resampler import PointSetResampler
from models.common import chamfer_distance_unit_sphere, coarsesGenerator
from utils.datasets import UpsampleDataset, PairedPatchDataset, pcl
from utils.transforms import *
from utils.misc import *
from utils.denoise import patch_based_upsample , patch_based_upsample_big
def addnoise (pcl , noise ):
    pcl = pcl + torch.randn_like(pcl) * noise
    data['noise_std'] = noise_std
    return data
def normalize(pcl, center=None, scale=None):
    if center is None:
        p_max = pcl.max(dim=0, keepdim=True)[0]
        p_min = pcl.min(dim=0, keepdim=True)[0]
        center = (p_max + p_min) / 2    # (1, 3)
    pcl = pcl - center
    if scale is None:
        scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
    pcl = pcl / scale
    return pcl, center, scale
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--ckpt_model' ,type = str , default = './ckpts/model_u.pth' )
    parser.add_argument('--ckpt_gen' ,type = str , default = './ckpts/upsmodel.pth' )
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Datasets and loaders
   
    val_dset = UpsampleDataset(
        root=config.dataset.dataset_root,
        dataset=config.dataset.dataset,
        split='test',
        resolution=config.dataset.resolutions[1],
        rate=16,
        noise_min = 1.5e-2,
        noise_max = 1.5e-2
    )
    
    print("resolution is %s" % (config.dataset.resolutions[1]))
    # Model
    model = PointSetResampler(config.model).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_model))
    upsampler = coarsesGenerator(rate = 16 ).to(args.device)
    upsampler.load_state_dict(torch.load(args.ckpt_gen))

    def validate(it):
        model.eval()
        upsampler.eval()
        all_clean = []
        all_denoised = []
        filepath = config.dataset.dataset+"_" + config.dataset.method+"_" + config.dataset.resolutions[0] +"_" + str(val_dset.rate) + "x"
        filepath = os.path.join( config.dataset.base_dir, filepath ) 
        os.makedirs(filepath, exist_ok=True)
        for i, data in enumerate(tqdm(val_dset, desc='Validate')):
            with torch.no_grad():
                pcl_noisy = data['ups'].to(args.device)
                pcl_low = data['original'].to(args.device)
                pcl_clean = data['gt'].to(args.device)
                pcl_denoised = upsampler(pcl_low.unsqueeze(0)).squeeze(0)
                file = os.path.join(filepath , data['name']+".xyz" )
                np.savetxt( file,pcl_denoised.detach().cpu().numpy() )
                all_clean.append(pcl_clean.unsqueeze(0))
                all_denoised.append(pcl_denoised.unsqueeze(0))
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)
        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        print('[Val] Iter %04d | CD %.8f  ' % (it, avg_chamfer))

    # Main loop
    try:
        cd_loss = validate(0)

    except KeyboardInterrupt:
        print('Terminating...')

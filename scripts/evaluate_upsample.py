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
from models.common import chamfer_distance_unit_sphere
from utils.datasets import UpsampleDataset, PairedPatchDataset
from utils.transforms import *
from utils.misc import *
from utils.evaluate import *
from utils.denoise import patch_based_upsample , patch_based_upsample_big

def denormalize ( pc , center , scale):
    return pc * scale + center
def validate_cd ( data , file):
    pcl_clean = data['gt'].to(args.device)
    name = data['name']
    pcl_denoised = np.loadtxt(os.path.join(file , name+".xyz"))
    pcl_denoised = torch.from_numpy(pcl_denoised).type(torch.FloatTensor).to(args.device)
    chamfer = chamfer_distance_unit_sphere(pcl_clean.unsqueeze(0),pcl_denoised.unsqueeze(0), batch_reduction='mean')[0].item()
    return chamfer
def validate_p2m(data , file):
    verts = data['meshes']['verts'].to(args.device)
    faces = data['meshes']['faces'].to(args.device)
    name = data['name']
    pcl_denoised = np.loadtxt(os.path.join(file , name+".xyz"))
    pcl_denoised = torch.from_numpy(pcl_denoised).type(torch.FloatTensor).to(args.device)
    p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_denoised,
                    verts=verts,
                    faces=faces
                ).mean()
    return p2f
def validate_hd(data , file):
    ref = data['gt'].to(args.device)
    name = data['name']
    gen = np.loadtxt(os.path.join(file , name+".xyz"))
    gen = torch.from_numpy(gen).type(torch.FloatTensor).to(args.device)
    hd = hausdorff_distance_unit_sphere(gen = gen.unsqueeze(0) , ref = ref.unsqueeze(0))
    return hd
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_root', type=str, default='./logs')
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
        rate = 16,
        noise_min = 0,
        noise_max = 0,
        need_mesh=True
    )
    print("resolution is %s" % (config.dataset.resolutions[1]))
    filepath = config.dataset.dataset+"_" + config.dataset.method+"_" + config.dataset.resolutions[0] +"_" + str(val_dset.rate) + "x"
    file = os.path.join( config.dataset.base_dir, filepath ) 
    print(file)
    def validate(it):
        global best
        avg_chamfer = 0 
        total_chamfer = 0 
        total_p2m = 0 
        total_hd = 0 
        for i, data in enumerate(tqdm(val_dset, desc='Validate')):
            total_p2m += validate_p2m(data,file)
            total_chamfer += validate_cd(data, file)
        avg_chamfer = total_chamfer / len(val_dset)
        avg_p2m = total_p2m / len(val_dset)
        print('[Val] noise %.6f | CD %.8f  ' % (config.dataset.val_noise, avg_chamfer))
        print('[Val] noise %.6f | P2M%.8f  ' % (config.dataset.val_noise, avg_p2m))

    # Main loop
    try:
        cd_loss = validate(0)

    except KeyboardInterrupt:
        print('Terminating...')

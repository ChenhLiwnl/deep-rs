import os
import argparse
import shutil
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from models.resampler import PointSetResampler
from models.common import chamfer_distance_unit_sphere
from utils.datasets import PointCloudDataset, PairedPatchDataset
from utils.transforms import *
from utils.misc import *
from utils.evaluate import *
from utils.denoise import patch_based_denoise
def rotate( pc, degree):
    degree = math.pi * degree / 180.0
    sin, cos = math.sin(degree), math.cos(degree)
    matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    #matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    #matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    matrix = torch.tensor(matrix).to("cuda")
    pc = torch.matmul(pc, matrix)
    return pc
def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
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
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Datasets and loaders
   
    val_dset = PointCloudDataset(
        root=config.dataset.dataset_root,
        dataset=config.dataset.dataset,
        split='test',
        input_root=config.dataset.input_root,
        resolution=config.dataset.resolutions[0],
        from_saved=False,
        need_mesh=True,
        transform = NormalizeUnitSphere() , 
    )
    
    print("resolution is %s" % (config.dataset.resolutions[0]))
    method = config.dataset.method
    resolution = config.dataset.resolutions[0]
    val_noise = str(config.dataset.val_noise)
    filepath = config.dataset.dataset+ "_"+method +"_"+ resolution +"_" + val_noise
    file = os.path.join(config.dataset.base_dir, filepath )
    #filepath = '/172.31.222.52_data/luost/denoisegf_data/results'
    file = os.path.join(filepath,file)
    print(file)
    def validate(it):
        global best
        avg_chamfer = 0 
        avg_p2f = 0 
        for i, data in enumerate(tqdm(val_dset, desc='Validate')):
            pcl_clean = data['pcl_clean'].to(args.device)
            verts = data['meshes']['verts'].to(args.device)
            faces = data['meshes']['faces'].to(args.device)
            name = data['name']
            pcl_denoised = np.loadtxt(os.path.join(file , name+".xyz"))
            pcl_denoised = torch.from_numpy(pcl_denoised).type(torch.FloatTensor).to(args.device)
            if val_noise == 'blensor':
                pcl_denoised = rotate(pcl_denoised , -90)  # blensor requires rotation, -90
            avg_p2f += point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_denoised,
                    verts=verts,
                    faces=faces
                ).mean()
            pcl_denoised, _ , _ = normalize(pcl_denoised,data['center'].cuda(),data['scale'].cuda()) 
            avg_chamfer += chamfer_distance_unit_sphere(pcl_clean.unsqueeze(0),pcl_denoised.unsqueeze(0), batch_reduction='mean')[0].item()
            
        avg_chamfer /= len(val_dset)
        avg_p2f /= len(val_dset)    
        print('[Val] noise %s | CD %.8f  ' % (config.dataset.val_noise, avg_chamfer))
        print('[Val] noise %s | P2M %.8f  ' % (config.dataset.val_noise, avg_p2f))

    # Main loop
    try:
        cd_loss = validate(0)

    except KeyboardInterrupt:
        print('Terminating...')

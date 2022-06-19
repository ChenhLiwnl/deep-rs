import os
import argparse
import shutil
from types import CodeType
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from models.resampler import PointSetResampler
from models.common import chamfer_distance_unit_sphere
from utils.datasets import PointCloudDataset, PairedPatchDataset
from utils.transforms import *
from utils.misc import *
from utils.denoise import patch_based_denoise , patch_based_denoise_big,denoise_large_pointcloud
from .validate_p2m import rotate
def denormalize ( pc , center , scale):
    return pc * scale + center
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--ckpt' , type = str ,default = './ckpts/denoise.pth' )
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
        resolution=config.dataset.resolutions[0],
        need_mesh=False,
        from_saved= True,
        noise_level = config.dataset.val_noise,
        input_root = config.dataset.input_root,
        transform = NormalizeUnitSphere() , 
    )
    print("resolution is %s" % (config.dataset.resolutions[0]))
    # Model
    model = PointSetResampler(config.model).to(args.device)
    model.load_state_dict(torch.load(args.ckpt))
    def validate(it):
        model.eval()
        filepath = config.dataset.dataset+"_" + config.dataset.method+"_" + config.dataset.resolutions[0] +"_" +str(config.dataset.val_noise)
        file = os.path.join( config.dataset.base_dir, filepath ) 
        os.makedirs(file, exist_ok=True)
        print(file)
        for i, data in enumerate(tqdm(val_dset, desc='Validate')):
            pcl_noisy = data['pcl_clean'].to(args.device) 
            center = data['center'].to(args.device)
            scale = data['scale'].to(args.device)
            pcl_denoised = patch_based_denoise(model, pcl_noisy)
            pcl_denoised = denormalize(pcl_denoised , center , scale)
            filename = os.path.join(file, data['name']+".xyz")
            np.savetxt( filename, pcl_denoised.cpu().numpy())
           
        print('[Val] noise %s , %s | Finished Saving ' % (config.dataset.val_noise, config.dataset.dataset))

    # Main loop
    try:
        cd_loss = validate(0)

    except KeyboardInterrupt:
        print('Terminating...')

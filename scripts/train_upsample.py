import os
import argparse
import shutil
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
import itertools
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from models.resampler import PointSetResampler
from models.common import chamfer_distance_unit_sphere,coarsesGenerator
from utils.datasets import UpsampleDataset, PairedUpsDataset
from utils.transforms import *
from utils.misc import *
from utils.denoise import patch_based_denoise, patch_based_upsample
import itertools
from utils.evaluate import *
def batch_pairwise_dist( x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x)
    diag_ind_y = torch.arange(0, num_points_y)
    if x.get_device() != -1:
        diag_ind_x = diag_ind_x.cuda(x.get_device())
        diag_ind_y = diag_ind_y.cuda(x.get_device())
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P

def get_cd_loss( ref , gen):
    P = batch_pairwise_dist(ref, gen)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins)
    return loss_1 + loss_2
if __name__ == '__main__':
    r = 5
    best = 10000
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

    # Logging
    log_dir = get_new_log_dir(args.log_root, prefix=config_name  + '_' +'retrain_ups_gen')
    ckpt_mgr = CheckpointManager(log_dir)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

# Datasets and loaders
    logger.info('Loading datasets')
    train_dset = PairedUpsDataset(
        datasets=[UpsampleDataset(
            root=config.dataset.dataset_root,
            dataset=config.dataset.dataset,
            rate = 4 , 
            split='train',
            resolution=resl,
            noise_min = 1e-2,
            noise_max = 2.5e-2,
        ) for resl in config.dataset.resolutions
        ],
        patch_size=config.dataset.patch_size,
        patch_ratio=4.2,
        on_the_fly=True 
    )

    val_dset = UpsampleDataset(
        root=config.dataset.dataset_root,
        dataset=config.dataset.dataset,
        split='test',
        resolution=config.dataset.resolutions[1],
        rate = 4,
        noise_min = 1.7e-2,
        noise_max = 1.7e-2,
        need_mesh=True
    )
    train_iter = get_data_iterator(DataLoader(train_dset, batch_size=config.train.train_batch_size, num_workers=config.train.num_workers, shuffle=True))

    # Model
    logger.info('Building model...')
    model = PointSetResampler(config.model).to(args.device)
    upsampler = coarsesGenerator(rate = 8 ).to(args.device)
    #logger.info(repr(model))
    #logger.info(repr(upsampler))

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        itertools.chain(upsampler.parameters(),model.parameters()),
        #model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        threshold=config.scheduler.threshold,
    )
    def train(it):
        batch = next(train_iter)
        pcl_low = batch['pcl_low'].to(args.device)
        pcl_noisy = batch['pcl_noisy'].to(args.device)
        pcl_gt = batch['pcl_gt'].to(args.device)
        # Reset grad and model state
        model.train()
        upsampler.train()
        optimizer.zero_grad()
        # Forward
       
        pcl_noisy = upsampler(pcl_low)
        cd_loss = get_cd_loss(
            gen = pcl_noisy , 
            ref = pcl_gt
        )
        
        vec_loss = model.get_loss_pc(
            p_query=pcl_noisy, 
            p_ctx=pcl_low,
            p_gt=pcl_gt,
            avg_knn=config.train.vec_avg_knn,
        )            
        loss = vec_loss
        loss += cd_loss
        # Backward
        loss.backward()
        optimizer.step()
        # Logging
        #logger.info('[Train] Iter %04d |cd Loss %.6f | vec Loss %.6f' % (
        #    it, cd_loss.item(),vec_loss.item()
        #))
        logger.info('[Train] Iter %04d || vec Loss %.6f' % (
            it,vec_loss.item()
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.flush()

    def validate(it):
        global best
        all_clean = []
        all_denoised = []
        #upsampler.eval()
        avg_p2m = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_dset, desc='Validate')):
                
                pcl_noisy = data['ups'].to(args.device)
                pcl_low = data['original'].to(args.device)
                pcl_clean = data['gt'].to(args.device)
                #if it != 1 :
                pcl_denoised = patch_based_upsample(model,pcl_low, pcl_noisy)
                #pcl_denoised = patch_based_denoise(model,pcl_noisy)
                avg_p2m += point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_denoised,
                    verts=data['meshes']['verts'].to(args.device),
                    faces=data['meshes']['faces'].to(args.device)
                ).mean()
                all_clean.append(pcl_clean.unsqueeze(0))
                all_denoised.append(pcl_denoised.unsqueeze(0))
        
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)
        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        avg_p2m /= len(val_dset)
        logger.info('[Val] Iter %04d | CD %.6f  P2M %.6f ' % (it, avg_chamfer , avg_p2m))
        writer.add_scalar('val/chamfer', avg_chamfer, it)
        writer.add_scalar('val/p2m', avg_p2m, it)
        if avg_p2m < best :
            best = avg_p2m
            torch.save(model.state_dict(),log_dir+'/model.pth')
            torch.save(upsampler.state_dict(),log_dir+'/upsmodel.pth')
        writer.flush()
        scheduler.step(avg_chamfer) 
        return avg_chamfer , avg_p2m

    # Main loop
    logger.info('Start training...')
    try:
        for it in range(1, config.train.max_iters+1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                cd_loss , _ = validate(it)
                '''
                opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
                ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
                '''

    except KeyboardInterrupt:
        logger.info('Terminating...')

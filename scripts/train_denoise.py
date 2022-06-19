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
from utils.denoise import patch_based_denoise


if __name__ == '__main__':
    best = 10000
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_root', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.log_root, prefix=config_name +'_'+str(config.dataset.val_noise)+'_')
    print(log_dir)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Datasets and loaders
    logger.info('Loading datasets')
    train_dset = PairedPatchDataset(
        datasets=[PointCloudDataset(
            root=config.dataset.dataset_root,
            dataset=config.dataset.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=config.dataset.noise_max, noise_std_min=config.dataset.noise_min, rotate=config.dataset.aug_rotate)
        ) for resl in config.dataset.resolutions
        ],
        patch_size=config.dataset.patch_size,
        patch_ratio=1.2,
        on_the_fly=True 
    )

    val_dset = PointCloudDataset(
        root=config.dataset.dataset_root,
        dataset=config.dataset.dataset,
        split='test',
        resolution=config.dataset.resolutions[2],
        transform=standard_train_transforms(noise_std_max=config.dataset.val_noise, noise_std_min=config.dataset.val_noise, rotate=False, scale_d=0),
    )
    train_iter = get_data_iterator(DataLoader(train_dset, batch_size=config.train.train_batch_size, num_workers=config.train.num_workers, shuffle=True))

    # Model
    logger.info('Building model...')
    model = PointSetResampler(config.model).to(args.device)
    logger.info(repr(model))

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
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
        if it % 10000 == 0 :
            print(it)
        batch = next(train_iter)
        p_noisy = batch['pcl_noisy'].to(args.device)
        p_clean = batch['pcl_clean'].to(args.device)

        # Reset grad and model state
        optimizer.zero_grad()
        model.train()

        # Forward
        loss = model.get_loss_pc(
            p_query=p_noisy, 
            p_ctx=p_noisy,
            p_gt=p_clean,
            avg_knn=config.train.vec_avg_knn,
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Logging
        logger.info('[Train] Iter %04d | Loss %.6f' % (
            it, loss.item(),
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.flush()

    def validate(it):
        global best
        all_clean = []
        all_denoised = []
        for i, data in enumerate(tqdm(val_dset, desc='Validate')):
            pcl_noisy = data['pcl_noisy'].to(args.device)
            pcl_clean = data['pcl_clean'].to(args.device)
            pcl_denoised = patch_based_denoise(model, pcl_noisy)
            all_clean.append(pcl_clean.unsqueeze(0))
            all_denoised.append(pcl_denoised.unsqueeze(0))
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)

        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
        writer.add_scalar('val/chamfer', avg_chamfer, it)
        writer.add_mesh('val/pcl', all_denoised[:2], global_step=it)
        if avg_chamfer < best :
            best = avg_chamfer
            torch.save(model.state_dict(),log_dir+'/model.pth')
        writer.flush()
        scheduler.step(avg_chamfer)

    # Main loop
    logger.info('Start training...')
    try:
        for it in range(1, config.train.max_iters+1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                cd_loss = validate(it)

    except KeyboardInterrupt:
        logger.info('Terminating...')

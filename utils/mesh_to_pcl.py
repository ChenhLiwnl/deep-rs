import os
import argparse
import multiprocessing as mp
import logging
import numpy as np
import point_cloud_utils as pcu
logging.basicConfig(level=logging.DEBUG)

'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./data/PUNet')
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
'''
DATASET_ROOT = '/172.31.222.52_data/chenhaolan/Mixed'
MESH_ROOT = os.path.join(DATASET_ROOT, 'meshes')
SAVE_ROOT = '/172.31.222.52_data/chenhaolan/Mixed/'
POINTCLOUD_ROOT = os.path.join(SAVE_ROOT, 'pointclouds')
RESOLUTIONS = [1024, 2048 , 4096, 8192 , 16384 , 32768 ,2048*12 ,2048*20 , 2048*32 , 2048*24]
SAMPLERS = ['poisson']
#SUBSETS = ['simple', 'medium', 'complex', 'train', 'test']
SUBSETS = ['train','test']
NUM_WORKERS = 32


def poisson_sample(v, f, n, num_points):
    pc, _ = pcu.sample_mesh_poisson_disk(v, f, n, num_points, use_geodesic_distance=True)
    if pc.shape[0] > num_points:
        pc = pc[:num_points, :]
    else:
        compl, _ = pcu.sample_mesh_random(v, f, n, num_points - pc.shape[0])
        # Notice: if (num_points - pc.shape[0]) == 1, sample_mesh_random will 
        #          return a tensor of size (3, ) but not (1, 3)
        compl = np.reshape(compl, [-1, 3])  
        pc = np.concatenate([pc, compl], axis=0)
    return pc


def random_sample(v, f, n, num_points):
    pc, _ = pcu.sample_mesh_random(v, f, n, num_points)
    return pc


def enum_configs():
    for subset in SUBSETS:
        for resolution in RESOLUTIONS:
            for sampler in SAMPLERS:
                yield (subset, resolution, sampler)


def enum_meshes():
    for subset, resolution, sampler in enum_configs():
        in_dir = os.path.join(MESH_ROOT, subset)
        out_dir = os.path.join(POINTCLOUD_ROOT, subset, '%d_%s' % (resolution, sampler))
        if not os.path.exists(in_dir):
            continue
        for fn in os.listdir(in_dir):
            if fn[-3:] == 'off':
                basename = fn[:-4]
                yield (subset, resolution, sampler,
                        os.path.join(in_dir, fn), 
                        os.path.join(out_dir, basename+'.xyz'))


def process(args):
    subset, resolution, sampler, in_file, out_file = args
    if os.path.exists(out_file):
        logging.info('Already exists: ' + in_file)
        return
    logging.info('Start processing: [%s,%d,%s] %s' % (subset, resolution, sampler, in_file))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    v, f, n = pcu.read_off(in_file)
    if sampler == 'poisson':
        pointcloud = poisson_sample(v, f, n, resolution)
    elif sampler == 'random':
        pointcloud = random_sample(v, f, n, resolution)
    else:
        raise ValueError('Unknown sampler: ' + sampler)
    np.savetxt(out_file, pointcloud, '%.6f')


if __name__ == '__main__':
    if NUM_WORKERS > 1:
        with mp.Pool(processes=NUM_WORKERS) as pool:
            pool.map(process, enum_meshes())
    else:
        for args in enum_meshes():
            process(args)
import numpy as np
import torch
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse
# from dataloader import *
from models.MobieN2_backbone import Model
import platform
from thop import profile
from thop import clever_format
import os

def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 448, 448).cuda()
    flow = torch.randn(1, 3, 448, 448).cuda()
    depth = torch.randn(1, 3, 448, 448).cuda()

    if device == 'cuda':
        model = model.cuda()

    model.eval()

    # #Calculate FLOPs and model parameters
    # macs, params = profile(model, inputs=(inputs,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print("MACs (FLOPs):", macs)
    # print("Number of parameters:", params)

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            # print()
            _,l,l1,l2,l3,l4 = model(inputs,flow,depth)
        if device == 'cuda':
            torch.cuda.synchronize()
        if idx > 10:
            time_spent.append(time.time() - start_time)
    
    #Calculate FLOPs and model parameters
    macs, params = profile(model, inputs=(inputs,flow,depth,))
    print(f"FLOPS: {macs / 1e9:.3f} GFLOPS")  # 转换为Giga FLOPS
    print(f"Params: {params / 1e6:.2f} M")  # 转换为百万
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs (FLOPs):", macs)
    print("Number of parameters:", params)
    print('Avg execution time (ms): %.4f, FPS:%d' % (np.mean(time_spent), 1*1//np.mean(time_spent) * 4))
    return 1*1//np.mean(time_spent)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.backends.cudnn.benchmark = True
    # mode = "test"

    # from model import MyModel
    device = torch.device('cuda:1')
    net = Model(mode = "test")
    computeTime(net)
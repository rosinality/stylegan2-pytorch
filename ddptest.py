import argparse
import os

import torch
from torch import nn
from torch.autograd import grad
from torch import distributed as dist
from torch.utils.data.sampler import Sampler

from model import Discriminator


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(42 + args.local_rank)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    print(n_gpu, args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()
    
    test_dist = True

    model = Discriminator(256).to('cuda')
    
    # if test_dist:
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )
    
    if not test_dist:
        model = model.module

    x = torch.randn(2, 3, 256, 256).to('cuda')
    x.requires_grad = True
    out = model(x)
    # print('model in', x)
    # print('model out', out)
    grad_in, = grad(out.sum(), inputs=x, create_graph=True)
    gp = grad_in.pow(2).view(2, -1).sum(1).mean()
    gp.mean().backward()
    
    print(model.module.convs[0][0].weight.grad.view(-1))
import argparse
import math
import random
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
    
except ImportError:
    wandb = None

from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return sampler.DistributedSampler(dataset, shuffle=shuffle)
    
    if shuffle:
        return sampler.RandomSampler(dataset)
    
    else:
        return sampler.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return (real_loss + fake_loss).mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.detach().mean() - mean_path_length
    )

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    
    pbar = range(args.iter)
    
    if get_rank() == 0:
        pbar = tqdm(pbar, dynamic_ncols=True)
    
    mean_path_length = 0
    
    d_loss_val = 0
    r1_loss = torch.tensor(0., device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0., device=device)
    loss_dict = {}
    
    sample_z = torch.randn(8 * 8, args.latent, device=device)
    
    for i in pbar:
        real_img = next(loader)
        real_img = real_img.to(device)
        
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        
        if args.mixing > 0 and random.random() < args.mixing:
            noise11, noise12, noise21, noise22 = torch.randn(4, args.batch, args.latent, device=device).chunk(4, 0)
            noise1 = [noise11.squeeze(0), noise12.squeeze(0)]
            noise2 = [noise21.squeeze(0), noise22.squeeze(0)]
            
        else:
            noise1, noise2 = torch.randn(2, args.batch, args.latent, device=device).chunk(2, 0)
            noise1 = [noise1.squeeze(0)]
            noise2 = [noise2.squeeze(0)]
        
        fake_img, _ = generator(noise1)
        fake_pred = discriminator(fake_img)
        
        d_regularize = i % args.d_reg_every == 0
        
        if d_regularize:
            real_img.requires_grad = True
        
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        
        loss_dict['d'] = d_loss
        
        discriminator.zero_grad()
        d_loss.backward(retain_graph=d_regularize)
        d_optim.step()
        
        if d_regularize:
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * d_regularize).backward()
            d_optim.step()
        
        loss_dict['r1'] = r1_loss
        
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        
        fake_img, latents = generator(noise2, return_latents=True)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        
        loss_dict['g'] = g_loss
        
        g_regularize = i % args.g_reg_every == 0
        
        generator.zero_grad()
        g_loss.backward(retain_graph=g_regularize)
        g_optim.step()
        
        if g_regularize:
            generator.zero_grad()
            path_loss, mean_path_length = g_path_regularize(fake_img, latents, mean_path_length)
            (args.path_regularize * path_loss * g_regularize).backward()
            mean_path_length_avg = reduce_sum(mean_path_length) / get_world_size()
            g_optim.step()
            
        loss_dict['path'] = path_loss
        
        accumulate(g_ema, generator.module)
        
        loss_reduced = reduce_loss_dict(loss_dict)
        
        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
            
        if get_rank() == 0:
            pbar.set_description((f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                                f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'))
            
            if wandb and args.wandb:
                wandb.log({'Generator': g_loss_val, 'Discriminator': d_loss_val, 'R1': r1_val,
                           'Path Length Regularization': path_loss_val, 'Mean Path Length': mean_path_length})
            
            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(sample, f'sample/{str(i).zfill(6)}.png', nrow=8, normalize=True, range=(-1, 1))
                    
            if i % 10000 == 0:
                torch.save({'g': generator.module.state_dict(), 'd': discriminator.module.state_dict(),
                            'g_ema': g_ema.state_dict(), 'g_optim': g_optim.state_dict(), 'd_optim': d_optim.state_dict()},
                           f'checkpoint/{str(i).zfill(6)}.pt')
        

if __name__ == '__main__':
    device = 'cuda'
    
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 * g_reg_ratio, 0.99 * g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 * d_reg_ratio, 0.99 * d_reg_ratio))
    
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    dataset = MultiResolutionDataset(args.path, transform)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed))
    
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='stylegan 2')
    
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
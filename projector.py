import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms


import lpips
from model import Generator

from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataSet
from tqdm import tqdm
from torch.backends import cudnn
from PIL import Image
import os.path as osp


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Image projector to the generator latent spaces")
    
    parser.add_argument("--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--size", type=int, default=256, help="output image sizes of the generator")
    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup",)
    parser.add_argument("--lr_rampdown",type=float,default=0.25,help="duration of the learning rate decay",)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--noise", type=float, default=0.05, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay",)
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--noise_regularize",type=float,default=1e5,help="weight of the noise regularization")
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument("--w_plus",action="store_true",help="allow to use distinct latent codes to each layers",)
    parser.add_argument("-b", "--batch_size", default=32)
    parser.add_argument("-img","--img_path", help="path to image folder to be projected")
    parser.add_argument("--gpu", default ='0', help="CUDA ID, e.g. 0 or 1,2") 
    parser.add_argument("--device", default='cuda',choices=['cuda','cpu']) 
    parser.add_argument("--n_mean_latent", default=1000) 
    parser.add_argument("-o","--output_path",default='projected_output') 
    parser.add_argument("--tqdm_off",action='store_true',default='turn on tqdm progressive bar off')


    args = parser.parse_args()
    w_flag = 'W' if not args.w_plus else 'W_PLUS'
    args.output_path = osp.join( args.output_path,f"projected_{w_flag}_{args.step}step_{args.size}_{osp.basename(args.ckpt).split('.')[0]}")
    args.output_feature_path = osp.join(args.output_path,'projected_latent_dict')
    args.output_img_path = osp.join(args.output_path,'inversed_imgs')
    os.makedirs(args.output_path,exist_ok=True)

    # TODO: this is not make use of multi-GPU yet
    if args.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # new add by gu
    cudnn.benchmark = True

    torch.cuda.set_device(torch.device('cuda',int(args.gpu[0])))
    
    # transform
    resize = min(args.size, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    print("Making Dataloader")
    print(f"Loading images from: {args.img_path}")
    my_dataset = CustomDataSet(args.img_path, transform=transform)
    dataloader = DataLoader(my_dataset , batch_size=args.batch_size, shuffle=False, 
                               num_workers=4, drop_last=True)
    print(f"Dataloader: total_imgs:{len(my_dataset.totals_imgs)} , batch_size {args.batch_size}")
    

    print("Init Generator")
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(args.device)

    n_mean_latent = args.n_mean_latent  # 1000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=args.device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=args.device.startswith("cuda")
    )
    print(f"the result will be saved at: {args.output_path}")
    print("start projection")
    print(f"total step : {args.step}")
    for i,batch in tqdm(enumerate(dataloader),disable=args.tqdm_off):
        fnames,imgs =batch
        imgs = imgs.to(args.device) if torch.cuda.device_count() >= 1 else imgs

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step),disable=args.tqdm_off)
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)
        img_ar = make_image(img_gen)
        
        # TODO: make it parallel (don't gain much), maybe using cache first and parallelly save it (shoud not be too big size)
        for i, fname in enumerate(fnames):
            noise_single = []
            for noise in noises:
                noise_single.append(noise[i : i + 1])
            # latent: "w",  img: "inversed image"
            projected_result = {
                "img": img_gen[i],
                "latent": latent_in[i], 
                "noise": noise_single,
            }
            
            img_name = f"{os.path.splitext(os.path.basename(fname))[0]}_reversed.png"
            pil_img = Image.fromarray(img_ar[i])
            pil_img.save(osp.join(args.output_img_path,img_name))
            feature_name = f"{os.path.splitext(os.path.basename(fname))[0]}_projected.pt"
            torch.save(projected_result, (args.output_feature_path,feature_name))

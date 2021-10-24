import argparse
import os
import os.path as osp
import torch
from torchvision import transforms
from torch.backends import cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # new add by gu
torch.cuda.set_device(torch.device('cuda',0))
cudnn.benchmark = True
from datasets.custom_dataset import CustomDataSet
from model import Generator
import torch.nn.functional as F
from collections import defaultdict


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




def init_styleGAN(img_size=256,ckpt='./checkpoint/550000.pt'):
    g_ema = Generator(img_size, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema.to('cuda')
    return g_ema



def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

def spherical_interpolation(x0, x1, alpha):
    theta = torch.acos(cos(x0, x1))   #torch.arccos(cos(x0, x1))
    a = torch.sin((1-alpha)*theta) / torch.sin(theta) * x0
    b = torch.sin(alpha*theta) / torch.sin(theta) * x1
    return a + b

def sqrt_interpolation(x0, x1, alpha):
    return ((1-alpha) * x0 + (alpha) * x1) / math.sqrt(alpha ** 2 + (1-alpha) ** 2)

def linear_interpolation(x0, x1, alpha):
    return ((1-alpha) * x0 + (alpha) * x1)    






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
def post_processing(img_gen):
    channel, height, width = img_gen.shape
    if height > 256:
        factor = height // 256
        img_gen = img_gen.reshape(channel, height // factor, factor, width // factor, factor)
        img_gen = img_gen.mean([3, 5])
    
    return img_gen 
def pair_interpolate(noises1,noises2,latent1,latent2,alphas,w_plus=False):
    global g_ema
    interpolated_imgs = [] 
    for alpha in alphas:
        interpolated_noises = []
        for noise1, noise2 in zip(noises1,noises2):
            interpolated_noises += [spherical_interpolation(noise1, noise2, alpha)]
        if w_plus:
            interpolated_latent = torch.stack([linear_interpolation(latent1_a,latent2_b,alpha) for latent1_a,latent2_b in zip(latent1,latent2) ])
        else: interpolated_latent = linear_interpolation(latent1,latent2,alpha)
        # print(interpolated_latent.shape,interpolated_latent[None, :].shape )
        interpolated_img, _ = g_ema([interpolated_latent[None, :]], input_is_latent=True, noise=interpolated_noises)
        interpolated_img = make_image(interpolated_img)
        interpolated_imgs +=  [interpolated_img[0]]
    return interpolated_imgs
def make_pair_interpolate(latent_pairs,alphas,w_plus=False):
    interpolated_pair_imgs = defaultdict(lambda: list())
    for i in latent_pairs:
        interpolated_pair_imgs[i] = pair_interpolate(latent_pairs[i]['a']['noise'],latent_pairs[i]['b']['noise'],latent_pairs[i]['a']['latent'],latent_pairs[i]['b']['latent'],alphas=alphas,w_plus=w_plus)
    return   interpolated_pair_imgs


def save_images(interpolated_pair_imgs,alphas,output_path='./output/',w_plus=False):
    w_flag = "w_plus" if w_plus else "w"
    output_path = osp.join(output_path,w_flag,f"{len(alphas)}alpha")
    os.makedirs(output_path,exist_ok=True)
    for i in interpolated_pair_imgs:
        pair_path = osp.join(output_path,f"pair{i}")
        os.makedirs(pair_path,exist_ok=True)
        for j,(img,alpha) in enumerate(zip(interpolated_pair_imgs[i],alphas)):
            img_path = osp.join(pair_path,f"{j}_alpha{alpha}_.png")
            pil_img = Image.fromarray(img)
            pil_img.save(img_path)
    print(f"saved images to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image projector to the generator latent spaces")
    parser.add_argument("-lp","--latent_path", default="./projected_output/interpolate_pairs/projected_W_1000step_256_550000/projected_latent_dict")
    parser.add_argument("--device", default='cuda',choices=['cuda','cpu']) 
    parser.add_argument("-ni", "--num_interpolate",type=int,default=7,help='alpha, frequency of interpolation between values 0-1')
    parser.add_argument("--w_plus",action="store_true",help="allow to use distinct latent codes to each layers",)
    parser.add_argument("-o","--output_path",default='output/celeb_pairs/') 
    parser.add_argument("-s","--img_size",type=int,default=256)
    parser.add_argument("-ckpt","--checkpoint",default='./checkpoint/550000.pt') 
    
    

    args = parser.parse_args()


    g_ema = init_styleGAN(args.img_size,args.checkpoint)

    alphas = np.linspace(0, 1, args.num_interpolate)
    
    latent_path =  args.latent_path
    latent_files = os.listdir(latent_path)
    latent_pairs = defaultdict(lambda: defaultdict(str))
    for latent_file in latent_files:
        img_id, img_tag =  latent_file.split("_")[:2]
        with open(osp.join(latent_path,latent_file),'rb') as f:
            latent_pairs[img_id][img_tag] = torch.load(f)
    print(f"there are {len(latent_files)} latent files")

    print("start interpolation")
    interpolated_pair_imgs = make_pair_interpolate(latent_pairs,alphas=alphas,w_plus=args.w_plus)
    print("finish interpolation")
    save_images(interpolated_pair_imgs,alphas=alphas,w_plus=args.w_plus,output_path=args.output_path)
    



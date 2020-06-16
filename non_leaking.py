import math

import torch
from torch.nn import functional as F


def translate_mat(t_x, t_y):
    batch = t_x.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate

    return mat


def rotate_mat(theta):
    batch = theta.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batch, 2, 2)
    mat[:, :2, :2] = rot

    return mat


def scale_mat(s_x, s_y):
    batch = s_x.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y

    return mat


def lognormal_sample(size, mean=0, std=1):
    return torch.empty(size).log_normal_(mean=mean, std=std)


def category_sample(size, categories):
    category = torch.tensor(categories)
    sample = torch.randint(high=len(categories), size=(size,))

    return category[sample]


def uniform_sample(size, low, high):
    return torch.empty(size).uniform_(low, high)


def normal_sample(size, mean=0, std=1):
    return torch.empty(size).normal_(mean, std)


def bernoulli_sample(size, p):
    return torch.empty(size).bernoulli_(p)


def random_affine_apply(p, transform, prev, eye):
    size = transform.shape[0]
    select = bernoulli_sample(size, p).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye

    return select_transform @ prev


def sample_affine(p, size, height, width):
    G = torch.eye(3).unsqueeze(0).repeat(size, 1, 1)
    eye = G

    # flip
    param = category_sample(size, (0, 1))
    Gc = scale_mat(1 - 2.0 * param, torch.ones(size))
    G = random_affine_apply(p, Gc, G, eye)
    # print('flip', G, scale_mat(1 - 2.0 * param, torch.ones(size)), sep='\n')

    # 90 rotate
    param = category_sample(size, (0, 3))
    Gc = rotate_mat(-math.pi / 2 * param)
    G = random_affine_apply(p, Gc, G, eye)
    # print('90 rotate', G, rotate_mat(-math.pi / 2 * param), sep='\n')

    # integer translate
    param = uniform_sample(size, -0.125, 0.125)
    param_height = torch.round(param * height) / height
    param_width = torch.round(param * width) / width
    Gc = translate_mat(param_width, param_height)
    G = random_affine_apply(p, Gc, G, eye)
    # print('integer translate', G, translate_mat(param_width, param_height), sep='\n')

    # isotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, param)
    G = random_affine_apply(p, Gc, G, eye)
    # print('isotropic scale', G, scale_mat(param, param), sep='\n')

    p_rot = 1 - math.sqrt(1 - p)

    # pre-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param)
    G = random_affine_apply(p_rot, Gc, G, eye)
    # print('pre-rotate', G, rotate_mat(-param), sep='\n')

    # anisotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, 1 / param)
    G = random_affine_apply(p, Gc, G, eye)
    # print('anisotropic scale', G, scale_mat(param, 1 / param), sep='\n')

    # post-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param)
    G = random_affine_apply(p_rot, Gc, G, eye)
    # print('post-rotate', G, rotate_mat(-param), sep='\n')

    # fractional translate
    param = normal_sample(size, std=0.125)
    Gc = translate_mat(param, param)
    G = random_affine_apply(p, Gc, G, eye)
    # print('fractional translate', G, translate_mat(param, param), sep='\n')

    return G


def apply_affine(img, G):
    grid = F.affine_grid(
        torch.inverse(G).to(img)[:, :2, :], img.shape, align_corners=False
    )
    img_affine = F.grid_sample(
        img, grid, mode="bilinear", align_corners=False, padding_mode="reflection"
    )

    return img_affine

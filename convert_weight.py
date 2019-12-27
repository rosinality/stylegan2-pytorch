import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np
from torchvision import utils

from model import Generator


def convert_modconv(vars, source_name, target_name, flip=False):
    weight = vars[source_name + '/weight'].value().eval()
    mod_weight = vars[source_name + '/mod_weight'].value().eval()
    mod_bias = vars[source_name + '/mod_bias'].value().eval()
    noise = vars[source_name + '/noise_strength'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.modulation.weight': mod_weight.transpose((1, 0)),
        'conv.modulation.bias': mod_bias + 1,
        'noise.weight': np.array([noise]),
        'activate.bias': bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + '.conv.weight'] = torch.flip(
            dic_torch[target_name + '.conv.weight'], [3, 4]
        )

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    weight = vars[source_name + '/weight'].value().eval()
    mod_weight = vars[source_name + '/mod_weight'].value().eval()
    mod_bias = vars[source_name + '/mod_bias'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.modulation.weight': mod_weight.transpose((1, 0)),
        'conv.modulation.bias': mod_bias + 1,
        'bias': bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    weight = vars[source_name + '/weight'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {'weight': weight.transpose((1, 0)), 'bias': bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        if k not in state_dict:
            raise KeyError(k + ' is not found')

        if v.shape != state_dict[k].shape:
            raise ValueError(f'Shape mismatch: {v.shape} vs {state_dict[k].shape}')

        state_dict[k] = v


def fill_statedict(state_dict, vars, size):
    log_size = int(math.log(size, 2))

    for i in range(8):
        update(state_dict, convert_dense(vars, f'G_mapping/Dense{i}', f'style.{i + 1}'))

    update(
        state_dict,
        {
            'input.input': torch.from_numpy(
                vars['G_synthesis/4x4/Const/const'].value().eval()
            )
        },
    )

    update(state_dict, convert_torgb(vars, 'G_synthesis/4x4/ToRGB', 'to_rgb1'))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(vars, f'G_synthesis/{reso}x{reso}/ToRGB', f'to_rgbs.{i}'),
        )

    update(state_dict, convert_modconv(vars, 'G_synthesis/4x4/Conv', 'conv1'))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                f'G_synthesis/{reso}x{reso}/Conv0_up',
                f'convs.{conv_i}',
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f'G_synthesis/{reso}x{reso}/Conv1', f'convs.{conv_i + 1}'
            ),
        )
        conv_i += 2

    return state_dict


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, required=True)
    parser.add_argument('path', metavar='PATH')

    args = parser.parse_args()

    sys.path.append(args.repo)

    from dnnlib import tflib

    tflib.init_tf()

    with open(args.path, 'rb') as f:
        _, _, g_ema = pickle.load(f)

    size = g_ema.output_shape[2]

    g = Generator(size, 512, 8)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, g_ema.vars, size)

    g.load_state_dict(state_dict)

    latent_avg = torch.from_numpy(g_ema.vars['dlatent_avg'].value().eval())

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save({'g_ema': state_dict, 'latent_avg': latent_avg}, name + '.pt')

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g.to(device)

    x = torch.randn(n_sample, 512).to(device)

    with torch.no_grad():
        img, _ = g([x], truncation=0.5, truncation_latent=latent_avg.to(device))

    utils.save_image(
        img, name + '.png', nrow=int(n_sample ** 0.5), normalize=True, range=(-1, 1)
    )

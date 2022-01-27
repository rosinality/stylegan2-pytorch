import argparse
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="automatically download the pretrained weight and save in the Pytorch default cache, necessary for offline training system")
    parser.add_argument("--arch", type=str, default='vgg16', help="backbone architecture to download",choices=list(model_urls.keys())) 
    
    args = parser.parse_args()

    arch = args.arch
    load_state_dict_from_url(model_urls[arch],progress=True)

    print(f"finish downloading pretrained weight {arch} and saving it in the Pytorch default cache")


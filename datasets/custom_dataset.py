from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform,completed_images=[],index_range=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        if index_range != None: 
            print(f"applying index range {index_range}")
            start_index = index_range[0]; end_index = index_range[1]
            all_imgs = all_imgs[start_index: end_index ]
        if len(completed_images) > 0:
            print(f"Continue projection from previous process that have finished projecting {len(completed_images)} images")
            if index_range != None: print(f"total in-range completion: {len(set(all_imgs).intersection(set(all_imgs)))} images")
            all_imgs = [img for img in all_imgs if img not in completed_images]
        if index_range != None:  print(f"total images to be process: {len(all_imgs)} images, within range [{index_range}]")
        else: print(f"total images to be process: {len(all_imgs)} images")
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        fname = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        img = self.transform(image)
        return fname,img

import os
import glob
import random
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class VideoTransform:
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BICUBIC):
        if not isinstance(scale, tuple) or not len(scale) == 2:
            raise ValueError('Scale should be a tuple with two elements.')
        
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img1, img2):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img1, self.scale, self.ratio)
        img1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2 = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)

        if random.random() < 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        img1 = self.normalize(transforms.ToTensor()(img1))
        img2 = self.normalize(transforms.ToTensor()(img2))

        return img1, img2


class YT18Dataset(Dataset):
    def __init__(self, input_size, root_dir):
        self.input_size = input_size
        self.transform = VideoTransform(size=input_size)
        
        # Get all frames from all videos
        self.frames = []
        subdirs = glob.glob(root_dir + '/*')
        for subdir in subdirs:
            frames_in_subdir = sorted(glob.glob(subdir + '/*'))
            # Append frame pairs (frame_i, frame_i+1) to the list
            for i in range(len(frames_in_subdir) - 1):
                self.frames.append((frames_in_subdir[i], frames_in_subdir[i+1]))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame1_path, frame2_path = self.frames[idx]
        
        frame1 = Image.open(frame1_path)
        frame2 = Image.open(frame2_path)
        
        return self.transform(frame1, frame2)


class K400Dataset(Dataset):
    def __init__(self, input_size, root_dir):
        self.root_dir = root_dir
        self.input_size = input_size
        self.transform = VideoTransform(size=input_size)

        selected_subdirs_file = "./data/selected_subdirs.txt"
        if os.path.isfile(selected_subdirs_file):
            with open(selected_subdirs_file, "r") as f:
                selected_subdirs = f.read().splitlines()
        else:
            self.subdirs = glob.glob(root_dir + '/*/*')
            num_selected_subdirs = len(self.subdirs) // 130
            selected_subdirs = random.sample(self.subdirs, num_selected_subdirs)
            with open(selected_subdirs_file, "w") as f:
                for subdir in selected_subdirs:
                    f.write(subdir + "\n")

        self.frame_pairs = []  # Store frame pairs for selected subdirs

        for subdir in selected_subdirs:
            frames = os.listdir(subdir)
            frames.sort()
            for i in range(len(frames) - 1):
                frame1 = os.path.join(subdir, frames[i])
                frame2 = os.path.join(subdir, frames[i + 1])
                self.frame_pairs.append((frame1, frame2))

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        frame1_path, frame2_path = self.frame_pairs[idx]

        frame1 = Image.open(frame1_path)
        frame2 = Image.open(frame2_path)

        return self.transform(frame1, frame2)

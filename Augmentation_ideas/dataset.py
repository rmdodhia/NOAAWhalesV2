# %%
import os
import numpy as np
import librosa
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import spectrogating

# %%


class Panama15(Dataset):

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.03), fillcolor=int(0.5 * 255)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    }

    def __init__(self, rootdir, dset='train', pre='plain'):
        if dset == 'val':
            dset = 'test'
        self.dset = dset
        self.pre = pre
        self.data = []
        self.t_mins = []
        self.labels = []
        self.data_root = rootdir
        self.ann_root = os.path.join(rootdir, 'lists')
        ann_dir = os.path.join(self.ann_root, '{}_15.txt'.format(dset))
        self.load_data(ann_dir) 

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '').split(' ')
                file_id = line_sp[0]
                tm = float(line_sp[1])
                lab = int(line_sp[2])
                self.data.append(file_id)
                self.t_mins.append(tm)
                self.labels.append(lab)

    def class_counts_cal(self):
        unique_labels, unique_counts = np.unique(self.labels, return_counts=True)
        return unique_labels, unique_counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_id = self.data[index]
        t_min = self.t_mins[index]
        label = self.labels[index]
        file_dir = os.path.join(self.data_root, 'wav', file_id)

        wav, sr = librosa.load(file_dir, sr=48000)

        if self.pre == 'denoise':
            sample = torch.tensor(denoise(wav, sr))
        elif self.pre == 'pcen':
            sample = torch.tensor(pcen(wav, sr))
        elif self.pre == 'plain':
            sample = torch.tensor(mel(wav, sr))

        sample = segment(sample, t_min).unsqueeze(0)

        if self.dset == 'train':
            sample = self.train_audio_transforms(sample)


        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB' if self.num_channels == 3 else 'L')

        if self.transform is not None:
            sample = self.transform(sample)


        return sample, label, file_dir

def denoise(wav, sr):
    noise = wav[0:1*sr]
    wav_dn = spectrogating.removeNoise(audio_clip=wav, noise_clip=noise,
        n_grad_freq=2,
        n_grad_time=4,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_std_thresh=2.5,
        prop_decrease=1.0,
        verbose=False,
        visual=False)
    mel_dn = librosa.feature.melspectrogram(y=wav_dn, sr=sr, n_mels=64)
    mel_dn_db = librosa.power_to_db(mel_dn, ref=np.max)
    return mel_dn_db

def pcen(wav, sr):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64)
    pcen = librosa.pcen(mel, sr=sr, gain=1.3, hop_length=512,
                        bias=2, power=0.3, time_constant=0.4, eps=1e-06, max_size=1)
    return pcen

def mel(wav, sr):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64)
    return mel

def segment(spec, t_min):
    sr_spec = spec.shape[1] / 60

    # if 60 - t_min >= 3:
    #     start = int(t_min * sr_spec)
    #     end = int((t_min + 3) * sr_spec)
    # else:
    #     start = int((60 - 3) * sr_spec)
    #     end = int(60 * sr_spec)
    if 60 - t_min >= 3:
        start = int(t_min * sr_spec)
        end = start + 281 
    else:
        end = int(60 * sr_spec)
        start = end - 281 

    return spec[:, start:end]

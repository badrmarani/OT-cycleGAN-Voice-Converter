from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader

import torchaudio

import os
import torch


class VCC2018(Dataset):
    def __init__(self, root1, root2, transform=None):
        super(VCC2018, self).__init__()
        self.root1 = root1
        self.root2 = root2
        self.transform = transform

        self.voices1 = os.listdir(root1)
        self.voices2 = os.listdir(root2)
        self.len_voices1 = len(self.voices1)
        self.len_voices2 = len(self.voices2)

    def __len__(self):
        return max(self.len_voices1, self.len_voices2)

    def __getitem__(self, index):
        voice1 = self.voices1[index % self.len_voices1]
        voice2 = self.voices2[index % self.len_voices2]
        voice1_path = os.path.join(self.root1, voice1)
        voice2_path = os.path.join(self.root2, voice2)

        voice1, sample_rate1 = torchaudio.load(voice1_path)
        voice2, sample_rate2 = torchaudio.load(voice2_path)

        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate1,
            n_mels=24,
            normalized=True,
        )
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate2,
            n_mels=24,
            normalized=True,
        )
        voice1 = transform(voice1)
        voice2 = transform(voice2)

        return [voice1, sample_rate1, voice2, sample_rate2]


def collate_fn(batch):
    voice1 = pack_sequence([items[0] for items in batch], enforce_sorted=False)
    sample_rate1 = pack_sequence([items[1] for items in batch], enforce_sorted=False)
    voice2 = pack_sequence([items[2] for items in batch], enforce_sorted=False)
    sample_rate2 = pack_sequence([items[3] for items in batch], enforce_sorted=False)
    return voice1.data, sample_rate1.data, voice2.data, sample_rate2.data


def dataset_loader(batch_size, type="train"):
    dataset = VCC2018(f"dataset/{type}/VCC2SF4", f"dataset/{type}/VCC2TM1")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=1,
        collate_fn=collate_fn,
    )

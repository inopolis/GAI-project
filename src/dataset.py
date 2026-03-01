import numpy as np
import torch
from torch.utils.data import Dataset


class CharBinDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int, dtype=np.uint16):
        self.data = np.fromfile(bin_path, dtype=dtype)
        self.block_size = block_size
        assert len(self.data) > block_size + 1, "Dataset too small for block_size."

    def __len__(self):
        # number of possible contiguous blocks
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1].astype(np.int64))
        y = torch.tensor(chunk[1:].astype(np.int64))
        return x, y
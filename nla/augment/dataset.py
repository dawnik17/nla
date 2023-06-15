from torch.utils.data import Dataset
import torch
import string
import numpy as np
from typing import List


class CDataset(Dataset):
    def __init__(self, data: List[str]):
        self.data = data

        chars = string.ascii_uppercase + string.digits + "./% "

        self.char2idx = {char: i + 1 for i, char in enumerate(chars)}
        self.char2idx.update({"": 0})  # padding
        self.idx2char = {i: char for char, i in self.char2idx.items()}

        words = set()
        self.max_length = 0

        for sentence in data:
            sentence_split = sentence.split()

            for word in sentence_split:
                self.max_length = max(self.max_length, len(word))
                words.add(word)

        self.words = list(words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        """
        "if x in self.char2idx" below is for homophones
        homophones module has it's own config i.e. different char2idx and max_length
        normally the characters we consider are: A-Z, 0-9, ./%
        for homophones we only consider A-Z
        """
        tensor = np.asarray(
            [self.char2idx[x] for x in self.words[idx] if x in self.char2idx]
        )
        tensor = torch.LongTensor(
            np.pad(
                tensor,
                (self.char2idx[""], self.max_length - len(tensor)),
                "constant",
            )
        )

        return tensor, self.words[idx]

from abc import ABCMeta, abstractmethod
from typing import List

from nla.augment.dataset import CDataset
from nla.augment.device import get_torch_device
from torch.utils.data import DataLoader


class DataLoad:
    def __init__(self, queries: List[str], batch_size: int):
        """_summary_

        Args:
            queries (List[str]): _description_
            batch_size (int): _description_
        """
        self.device = get_torch_device()
        self.dataset = CDataset(data=queries)

        # dataloader
        dataloader_args = dict(shuffle=False, batch_size=batch_size, num_workers=0)
        self.dataloader = DataLoader(self.dataset, pin_memory=True, **dataloader_args)


class IAugment(metaclass=ABCMeta):
    def __init__(self, dataload: DataLoad):
        self.dataset = dataload.dataset
        self.dataloader = dataload.dataloader
        self.device = dataload.device

    @abstractmethod
    def compute(self):
        pass

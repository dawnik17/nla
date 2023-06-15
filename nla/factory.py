import os
import re
from collections import defaultdict
from typing import Callable, List

import yaml

from nla.augment import DataLoad
from nla.homophone import Homophone
from nla.ngram import EdgeNGrams
from nla.typo import Typo
from nla.word_boundary import WordBoundary
from nla.wrappers import SentenceWrapper, EnsembleWrapper


class Factory:
    def __init__(self, queries: List[str], batch_size: int):
        self.dataload = DataLoad(queries, batch_size)
        self.factory = defaultdict(lambda: dict())

    def _set_aug(self, aug):
        if aug not in self.factory:
            self.factory[aug] = self._word_aug(aug)(self.dataload)

    def _word_aug(self, aug):
        if aug == "ngrams":
            return EdgeNGrams

        elif aug == "word_boundary":
            return WordBoundary

        elif aug == "typo":
            return Typo

        elif aug == "homophone":
            return Homophone

        else:
            raise ValueError("Invalid augmentation")

    def __call__(self, aug: str, mode: str = "word") -> Callable:
        if aug == "ensemble" or mode == "ensemble":
            config_path = f"{os.path.abspath(os.getcwd())}/config.yml"

            with open(config_path) as file:
                config = yaml.full_load(file)
                aug_set = {re.sub("\d+", "", aug_) for aug_ in config}

            augs = dict()

            for aug_ in aug_set:
                self._set_aug(aug_)
                augs.update({aug_: self.factory[aug_]})

            return EnsembleWrapper(augs)

        else:
            self._set_aug(aug)

        return (
            self.factory[aug]
            if mode == "word"
            else SentenceWrapper({aug: self.factory[aug]})
        )

import bz2
import os
import re
from random import choices, random, sample
from typing import Dict, List, Tuple

import _pickle as cPickle
import yaml
from nla.augment import IAugment
from tqdm import tqdm


class EnsembleWrapper:
    def __init__(self, word_aug: Dict[str, IAugment]):
        """This is a sentence wrapper around word augmentation
        to provide augmentations for sentences.

        Args:
            word_aug (IAugment): Word Augmentation Object
        """
        self.word_aug = word_aug

        with open(f"{os.path.abspath(os.getcwd())}/config.yml") as file:
            self.config = yaml.full_load(file)

    def compute(self, **kwargs):
        extra_args = {
            "count_per_query": kwargs["count_per_query"],
            "progress_bar": kwargs["progress_bar"],
        }

        ratio = kwargs["ratio"]
        del kwargs["ratio"]

        aug_select_percent = []
        all_word_augs = []
        percentage = 0

        for aug_name, config in self.config.items():
            aug_name = re.sub("\d+", "", aug_name)

            # percentage
            percentage += config["percentage"]
            del config["percentage"]

            # check if precomputed data available
            precompute = config["precompute"]
            precompute_path = config["precompute_path"]
            del config["precompute"]
            del config["precompute_path"]

            config.update(extra_args)

            # get word augmentations from precomputed data
            if precompute and precompute_path:
                word_aug = cPickle.load(bz2.BZ2File(precompute_path, "r"))
            # real time augmentations
            else:
                word_aug = self.word_aug[aug_name].compute(**config)
                word_aug = self.tuple2dict(word_aug)

            all_word_augs.append(word_aug)
            aug_select_percent.append(percentage)

        return self.sample(
            data=list(self.word_aug.values())[0].dataset.data,
            list_of_aug_dict=all_word_augs,
            aug_select_percent=aug_select_percent,
            ratio=ratio,
            progress_bar=kwargs["progress_bar"],
            count_per_query=kwargs["count_per_query"],
        )

    @staticmethod
    def sample(
        data: List[str],
        list_of_aug_dict: List[Dict],
        aug_select_percent: List[float],
        count_per_query: int,
        ratio: float = 1,
        progress_bar: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Args:
            data (List[str]): List of sentences
            list_of_aug_dict (List[Dict]): List of Word level Aug dictionary. Example - [{"word": ["aug1", "aug2"], ..}, ..]
            count_per_query (int): per sentence count in the output
            ratio (float, optional): Fraction of words to touch per sentence. Defaults to 1.
            progress_bar (bool, optional): Show progress bar. Defaults to True.

        Returns:
            List[Tuple[str, str]]: [(Augmented Sentence i, Sentence i), (Augmented Sentence j, Sentence j)]
        """
        pbar = tqdm(data, disable=not progress_bar)
        pbar.set_description("Ensemble: ")

        result = []

        for sent in pbar:
            out = []

            for word in sent.split():
                aug_dict = choices(
                    list_of_aug_dict, k=1, cum_weights=aug_select_percent
                )[0]

                if word in aug_dict and random() <= ratio:
                    out.append(
                        sample(
                            aug_dict[word], k=min(count_per_query, len(aug_dict[word]))
                        )
                    )

                else:
                    out.append([word] * count_per_query)

            result.extend([(" ".join(o), sent) for o in zip(*out)])

        return result

    @staticmethod
    def tuple2dict(tuple_: Tuple[str, str]):
        dict_ = {}

        for aug, word in tuple_:
            dict_.setdefault(word, []).append(aug)

        return dict_

    def __call__(
        self,
        count_per_query: int,
        ratio: float = 1,
        progress_bar: bool = True,
        precompute: bool = False,
        **kwargs,
    ):
        kwargs.update(
            {
                "progress_bar": progress_bar,
                "ratio": ratio,
                "count_per_query": count_per_query,
                "precompute": precompute,
            }
        )
        return self.compute(**kwargs)

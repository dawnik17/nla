import bz2
from random import random, sample
from typing import Dict, List, Tuple
import os
import _pickle as cPickle
from nla.augment import IAugment
from tqdm import tqdm


class SentenceWrapper:
    def __init__(self, word_aug: Dict[str, IAugment]):
        """This is a sentence wrapper around word augmentation
        to provide augmentations for sentences.

        Args:
            word_aug (IAugment): Word Augmentation Object
        """
        for aug_name, word_aug in word_aug.items():
            self.aug_name = aug_name
            self.word_aug = word_aug

        self.precompute_path = (
            f"{os.path.abspath(os.getcwd())}/nla/precompute/{self.aug_name}.pbz2"
        )

        # we initialise this as None and only load it when called for
        # (lazy loading) (compute method)
        self.precomputed_data = None

    def compute(self, **kwargs):
        ratio = kwargs["ratio"]
        del kwargs["ratio"]

        precompute = kwargs["precompute"]
        del kwargs["precompute"]

        if precompute:
            if not self.precomputed_data:
                self.precomputed_data = cPickle.load(
                    bz2.BZ2File(self.precompute_path, "r")
                )

            word_aug = self.precomputed_data
        else:
            word_aug = self.word_aug(**kwargs)
            word_aug = self.tuple2dict(word_aug)

        return self.sample(
            data=self.word_aug.dataset.data,
            aug_dict=word_aug,
            ratio=ratio,
            progress_bar=kwargs["progress_bar"],
            count_per_query=kwargs["count_per_query"],
        )

    def sample(
        self,
        data: List,
        aug_dict: Dict,
        count_per_query: int,
        ratio: float = 1,
        progress_bar: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        result = []

        for sent in data:
            out = []

            for word in sent.split():
                if word in aug_dict and random.random() <= ratio:
                    out.append(
                        random.sample(
                            aug_dict[word], k=min(count_per_query, len(aug_dict[word]))
                        )
                    )

                else:
                    out.append([word] * count_per_query)

            result.extend([(" ".join(o), sent) for o in zip(*out)])

        Args:
            data (List): List of sentences
            aug_dict (Dict): Word level augmentations. Example - {"word": ["aug1", "aug2"]}
            count_per_query (int): _description_
            ratio (float, optional): Fraction of words to touch per sentence. Defaults to 1.
            progress_bar (bool, optional): Show progress bar. Defaults to True.

        Returns:
            List[Tuple[str, str]]: [(Augmented Sentence i, Sentence i), (Augmented Sentence j, Sentence j)]
        """
        pbar = tqdm(data, disable=not progress_bar)
        pbar.set_description(f"Sentence: {self.aug_name}")

        return [
            (" ".join(o), sent)
            for sent in pbar
            for o in zip(
                *[
                    sample(aug_dict[word], k=min(count_per_query, len(aug_dict[word])))
                    if random() <= ratio and word in aug_dict
                    else [word] * count_per_query
                    for word in sent.split()
                ]
            )
        ]

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

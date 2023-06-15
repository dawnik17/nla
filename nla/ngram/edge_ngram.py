from typing import Callable, List

import numpy as np
import torch
from nla.augment import IAugment, DataLoad
from tqdm import tqdm


class EdgeNGrams(IAugment):
    def __init__(self, dataload: DataLoad):
        """_summary_

        Args:
            dataload (DataLoad): _description_
        """
        super().__init__(dataload=dataload)

    def batch_compute(
        self,
        queries: np.array,
        query_tensors: torch.Tensor,
        degree: int,
        threshold: int,
    ) -> List[tuple]:
        """Get all edge ngrams of the batch array of strings

        Args:
            queries (np.array): array of string queries to be processed
            query_tensors (torch.Tensor): tensors of those queries (2D)
            degree (int): maximum number of characters to be removed per word
            threshold (int): minimum length of a word post edging ngrams

        Returns:
            List[tuple]: List of ngrams
        """
        # get number of characters in each word/tensor
        nonzeros = (query_tensors != 0).sum(-1)

        ngrams = []
        words = []

        for i in range(threshold, self.dataset.max_length):
            # start slicing
            cut = query_tensors[:, :i].clone()

            # keep a track of number of charaters being removed
            # only keep those sliced words which meet our condition (degree >= chars_removed)
            cut_nonzero = (cut != 0).sum(-1)
            chars_removed = nonzeros - cut_nonzero
            degree_threshold = degree >= chars_removed

            cut = cut[degree_threshold].detach().cpu().numpy()
            indx = (degree_threshold).nonzero(as_tuple=True)[0].detach().cpu().numpy()

            if len(indx) == 0:
                del cut
                del indx
                continue

            # convert indices back to chars
            words.extend(queries[indx])
            ngrams.extend(
                ["".join(num) for num in (np.vectorize(self.dataset.idx2char.get)(cut))]
            )

            del cut
            del indx

        return list(set(zip(ngrams, words)))

    def compute(
        self,
        degree: int,
        threshold: int,
        count_per_query: int,
        progress_bar: bool = True,
    ) -> List[tuple]:
        """Get all edge ngrams for the queries

        Args:
            degree (int): maximum number of characters to be removed per word
            threshold (int): minimum length of a word post edging ngrams

        Returns:
            List[tuple]: list of ngrams
        """
        pbar = tqdm(self.dataloader, disable=not progress_bar)
        pbar.set_description(
            f"Ngrams: count_per_query {count_per_query}, threshold {threshold}, degree {degree}"
        )

        output = set()

        for _, (tensor, queries) in enumerate(pbar, 0):
            tensor = tensor.to(self.device)
            queries = np.array(queries)

            output.update(self.batch_compute(queries, tensor, degree, threshold))

        torch.cuda.empty_cache()
        return list(output)

    def __call__(
        self, degree: int, threshold: int, count_per_query: int, progress_bar: bool = True
    ) -> Callable:
        """Get all edge ngrams for the queries

        Args:
            degree (int): maximum number of characters to be removed per word
            threshold (int): minimum length of a word post edging ngrams

        Returns:
            Callable: compute method
        """
        return self.compute(
            degree=degree, threshold=threshold, count_per_query=count_per_query, progress_bar=progress_bar
        )


if __name__ == "__main__":
    from nla.augment import Augment

    queries = ["CROCIN", "TABLET", "PARACETAMOL", "IVERMECTIN", "CIPLA"]

    aug = Augment(queries=queries, batch_size=3)
    edgen = aug(aug="ngrams")

    degree = 3
    threshold = 2
    print(edgen(degree, threshold))

import re
from typing import Callable, List

import numpy as np
import torch
from nla.augment import IAugment, DataLoad
from nla.typo.engine import TypoEngine
from tqdm import tqdm


class WordBoundary(IAugment):
    def __init__(self, dataload: DataLoad):
        """_summary_

        Args:
            dataload (DataLoad): _description_
        """
        super().__init__(dataload=dataload)

        # space before 4rd index i.e. space after 4 characters. Ex - mama earth.
        self.human_tendency_idx = 4
        self.engine = TypoEngine()

    def batch_compute(
        self,
        queries: np.array,
        query_tensors: torch.Tensor,
        degree: int,
        count_per_query: int,
    ) -> List[tuple]:

        words = []
        output = []

        nonzeros = (query_tensors != 0).sum(-1).cpu().numpy()

        query_tensors = query_tensors.repeat((count_per_query, 1, 1))
        queries = np.tile(queries, count_per_query)

        for _ in range(degree):
            rng = np.random.default_rng()

            # index of each row/word to touch
            indices = rng.integers(
                low=1, high=nonzeros + 1, size=(count_per_query, len(nonzeros))
            )
            indices = torch.from_numpy(indices).to(self.device)

            # insert space at these indices
            query_tensors = self.engine.insert3D(
                query_tensors,
                indices,
                torch.full(
                    indices.size(), self.dataset.char2idx[" "], device=self.device
                ),
            )

            # convert indices back to chars
            output.extend(
                [
                    re.sub(" +", " ", "".join(num)).strip()
                    for query_tensor in query_tensors
                    for num in (
                        np.vectorize(self.dataset.idx2char.get)(
                            query_tensor.cpu().numpy()
                        )
                    )
                ]
            )
            words.extend(queries)

        del query_tensors
        del indices

        return list(set(zip(output, words)))

    def human_compute(
        self, queries: np.array, query_tensors: torch.Tensor
    ) -> List[tuple]:
        words = []
        wb = []

        nonzeros = (query_tensors != 0).sum(-1)

        # human tendency word boundary
        threshold = nonzeros > self.human_tendency_idx

        # get indices and tensor rows for the valid thresholds
        indx = threshold.nonzero(as_tuple=True)[0].detach().cpu().numpy()
        cut = query_tensors[threshold]

        # vectorized version of self.engine.insert2D
        # (since we have to insert at the same index for all the tensor rows)
        cut = torch.cat(
            [
                cut[:, : self.human_tendency_idx],
                torch.full(
                    (cut.shape[0], 1), self.dataset.char2idx[" "], device=self.device
                ),
                cut[:, self.human_tendency_idx :],
            ],
            dim=-1,
        )

        if cut.size(0) == 0:
            del cut
            del indx
            return list()

        # convert indices back to query and chars
        words.extend(queries[indx])
        wb.extend(
            [
                "".join(num).strip()
                for num in (np.vectorize(self.dataset.idx2char.get)(cut.cpu().numpy()))
            ]
        )

        del cut
        del indx

        return list(set(zip(wb, words)))

    def compute(
        self, degree: int, count_per_query: int, progress_bar: bool = True
    ) -> List[tuple]:
        """Get a list of space errored terms for the queries

        Args:
            degree (int): maximum number of spaces to be introduced per word
            count_per_query (int): result count per query

        Returns:
            List[tuple]: list of space errored words
        """
        pbar = tqdm(self.dataloader, disable=not progress_bar)
        pbar.set_description(
            f"Word Boundary: count_per_query {count_per_query}, degree {degree}"
        )

        output = set()

        count_per_query = max(1, count_per_query // degree)

        for i, (tensor, queries) in enumerate(pbar, 0):
            tensor = tensor.to(self.device)
            queries = np.array(queries)

            output.update(
                self.batch_compute(queries, tensor.clone(), degree, count_per_query)
            )

            if i == 0:
                output.update(self.human_compute(queries, tensor.clone()))

        torch.cuda.empty_cache()
        return list(output)

    def __call__(
        self, degree: int, count_per_query: int, progress_bar: bool = True
    ) -> Callable:
        """Get a list of space errored terms for the queries

        Args:
            degree (int): maximum number of spaces to be introduced per word
            count_per_query (int): result count per query

        Returns:
            Callable: compute method
        """
        return self.compute(
            degree=degree, count_per_query=count_per_query, progress_bar=progress_bar
        )


if __name__ == "__main__":
    from nla.augment import Augment

    queries = ["CROCIN", "TABLET", "PARACETAMOL", "IVERMECTI"]
    degree = 2
    per_query_count = 4

    aug = Augment(queries=queries, batch_size=2)
    wb = aug(aug="word_boundary")
    print(wb(degree, per_query_count))

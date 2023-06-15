import json
import logging
import random
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import torch
from nla.augment import IAugment, DataLoad
from nla.typo.engine import TypoEngine
from tqdm import tqdm


class Typo(IAugment):
    def __init__(self, dataload: DataLoad):
        """_summary_

        Args:
            dataload (DataLoad): _description_
        """
        super().__init__(dataload=dataload)
        self.engine = TypoEngine()

        path = Path(__file__).parents[0] / "keyboard.json"
        self.keyboard = json.load(open(path))

        self.keyboard = {
            self.dataset.char2idx[key]: [self.dataset.char2idx[v] for v in value]
            for key, value in self.keyboard.items()
        }

        self.valid_touch_modes = ["insert", "swap", "replace", "delete", "random"]
        self.valid_aug_modes = ["keyboard", "random"]

    def _get_indices_to_touch(
        self, low: Union[int, np.array], high: np.array, count_per_query: int
    ) -> np.array:
        """Get indices for each row/word of the tensor that we wish to touch

        Args:
            low (Union[int, np.array]): minimum value (included)
            high (np.array): maximum value (excluded)

        Returns:
            np.array: array of indices to touch
        """
        rng = np.random.default_rng()

        # index of each row/word to touch
        return rng.integers(low=low, high=high, size=(count_per_query, len(high)))

    def _get_values_to_touch_with(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        aug_mode: str,
        count_per_query: int,
    ) -> List[int]:
        """Get values we wish to touch (insert/replace) with at the given indices,
        the indices fetched using _get_indices_to_touch method.

        Args:
            tensor (torch.Tensor): base tensor
            indices (torch.Tensor): indices we wish to touch
            aug_mode (str): aug_mode - ("keyboard" or "random")

        Raises:
            Exception: "argument aug_mode can only be 'keyboard' or 'random'"

        Returns:
            List[int]: values we wish to touch with (insert/replace)
        """
        if aug_mode == "keyboard":
            values = (
                torch.gather(tensor, dim=2, index=indices.unsqueeze(dim=2))
                .squeeze(-1)
                .tolist()
            )
            return torch.tensor(
                [
                    [
                        random.choice(self.keyboard[value_])
                        if value_ in self.keyboard
                        else value_
                        for value_ in value
                    ]
                    for value in values
                ],
                device=self.device,
            )

        elif aug_mode == "random":
            return torch.tensor(
                [
                    np.random.choice(list(self.keyboard), indices.size(1)).tolist()
                    for _ in range(count_per_query)
                ],
                device=self.device,
            )

        else:
            raise Exception("argument aug_mode can only be 'keyboard' or 'random'")

    # "insert" touch mode
    def insert(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        aug_mode: str,
        count_per_query: int,
    ) -> torch.Tensor:
        """Insert values based on the 'aug_mode' at the given 'indices' in the given 'tensor'

        Args:
            tensor (torch.Tensor): tensor (2D) to insert values in
            indices (torch.Tensor): indices of each row in the tensor where we wish to insert
            aug_mode (str): aug_mode - ("keyboard" or "random")

        Returns:
            torch.Tensor: _description_
        """
        values = self._get_values_to_touch_with(
            tensor, indices, aug_mode, count_per_query
        )
        return self.engine.insert3D(tensor, indices, values)

    # "replace" touch mode
    def replace(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        aug_mode: str,
        count_per_query: int,
    ) -> torch.Tensor:
        """Replace values based on the 'aug_mode' at the given 'indices' in the given 'tensor'

        Args:
            tensor (torch.Tensor): tensor (2D) to replace values in
            indices (torch.Tensor): indices of each row in the tensor which we wish to replace
            aug_mode (str): aug_mode - ("keyboard" or "random")_

        Returns:
            torch.Tensor: _description_
        """
        values = self._get_values_to_touch_with(
            tensor, indices, aug_mode, count_per_query
        )
        return self.engine.replace3D(tensor, indices, values)

    # "swap" touch mode
    def swap(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        aug_mode: str,
        count_per_query: int,
    ) -> torch.Tensor:
        """Swap values at the given 'indices' in the given 'tensor' based on the 'aug_mode'

        Args:
            tensor (torch.Tensor): tensor (2D) to swap values in
            indices (torch.Tensor): indices of each row in the tensor which we wish to swap
            aug_mode (str): aug_mode - ("keyboard" or "random")

        Returns:
            torch.Tensor: _description_
        """
        if aug_mode == "keyboard":
            tensor = self.replace(tensor, indices, aug_mode, count_per_query)

        return self.engine.swap3D(tensor, indices_x=indices, indices_y=indices + 1)

    # "delete" touch mode
    def delete(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        aug_mode: str,
        count_per_query: int,
    ) -> torch.Tensor:
        """Delete values at the given 'indices' in the given 'tensor'.

        Args:
            tensor (torch.Tensor): tensor (torch.Tensor): tensor (2D) to delete values from
            indices (torch.Tensor): indices of each row in the tensor which we wish to delete
            aug_mode (str): aug_mode - ("keyboard" or "random")

        Returns:
            torch.Tensor: _description_
        """
        if aug_mode == "keyboard":
            logging.info(
                "'delete' touch mode for the aug mode 'keyboard' doesn't make sense. \
                We'll be returning default delete results"
            )
        return self.engine.delete3D(
            tensor, indices, torch.tensor(self.dataset.char2idx[""], device=self.device)
        )

    # "random" touch mode
    def random(
        self,
        tensor: torch.Tensor,
        indices: np.array,
        aug_mode: str,
        count_per_query: int,
    ) -> torch.Tensor:
        touch_mode = "random"

        while touch_mode == "random":
            touch_mode = random.choice(self.valid_touch_modes)

        return getattr(self, touch_mode)(tensor, indices, aug_mode, count_per_query)

    def batch_compute(
        self,
        queries: np.array,
        query_tensors: torch.Tensor,
        degree: int,
        touch_mode: str,
        aug_mode: str,
        count_per_query: int,
    ) -> List[tuple]:
        """_summary_

        Args:
            queries (np.array): _description_
            query_tensors (torch.Tensor): _description_
            degree (int): _description_
            touch_mode (str): _description_
            aug_mode (str): _description_

        Raises:
            Exception: _description_

        Returns:
            List[tuple]: _description_
        """
        # select engine based on the touch mode
        engine = getattr(self, touch_mode)

        # get number of characters in each word/tensor
        nonzeros = (query_tensors != 0).sum(-1).cpu().numpy()

        query_tensors = query_tensors.repeat((count_per_query, 1, 1))
        queries = np.tile(queries, count_per_query)

        output = []
        words = []

        for _ in range(degree):
            # index of each row/word to touch
            indices = self._get_indices_to_touch(
                low=0, high=nonzeros - 1, count_per_query=count_per_query
            )
            indices = torch.from_numpy(indices).to(self.device)
            query_tensors = engine(query_tensors, indices, aug_mode, count_per_query)

            # convert indices back to chars
            output.extend(
                [
                    "".join(num)
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

    def compute(
        self,
        degree: int,
        count_per_query: int,
        touch_mode: str,
        aug_mode: str,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Introduce Typographical errors in a list of words.

        Args:
            degree (int): how many max operations do wish to do on one string
            count_per_query (int): result count per query
            touch_mode (str): "insert"/"replace"/"swap"/"delete"/"random"
            aug_mode (str): "keyboard"/"random"

        Returns:
            torch.Tensor: list of typographically augmented words
        """
        assert touch_mode in self.valid_touch_modes
        assert aug_mode in self.valid_aug_modes

        pbar = tqdm(self.dataloader, disable=not progress_bar)
        pbar.set_description(
            f"Typo: {touch_mode} touch_mode, {aug_mode} aug_mode, {count_per_query} count_per_query, {degree} degree"
        )

        output = set()

        count_per_query = max(1, count_per_query // degree)

        for _, (tensor, queries) in enumerate(pbar, 0):
            tensor = tensor.to(self.device)
            queries = np.array(queries)

            output.update(
                self.batch_compute(
                    queries=queries,
                    query_tensors=tensor.clone(),
                    degree=degree,
                    touch_mode=touch_mode,
                    aug_mode=aug_mode,
                    count_per_query=count_per_query,
                )
            )

        torch.cuda.empty_cache()
        return list(output)

    def __call__(
        self,
        degree: int,
        count_per_query: int,
        touch_mode: str,
        aug_mode: str,
        progress_bar: bool = True,
    ) -> Callable:
        """Introduce Typographical errors in a list of words.

        Args:
            degree (int): how many max operations do wish to do on one string
            count_per_query (int): result count per query
            touch_mode (str): "insert"/"replace"/"swap"/"delete"/"random"
            aug_mode (str): "keyboard"/"random"

        Returns:
            Callable: compute method
        """
        return self.compute(
            degree=degree,
            count_per_query=count_per_query,
            touch_mode=touch_mode,
            aug_mode=aug_mode,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    from nla.augment import Augment

    queries = ["CROCIN", "TABLET", "PARACETAMOL", "IVERMECTI"]
    degree = 2
    per_query_count = 4
    touch_mode = "insert"
    aug_mode = "keyboard"

    aug = Augment(queries=queries, batch_size=2)
    typo = aug(aug="typo")
    print(
        typo(
            degree=degree,
            count_per_query=per_query_count,
            touch_mode=touch_mode,
            aug_mode=aug_mode,
        )
    )

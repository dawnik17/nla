import logging
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from nla.augment import DataLoad, IAugment
from nla.homophone.model import Transformer
from torch.autograd import Variable
from tqdm import tqdm

REGEXP = re.compile(r"[^A-Z]")


class Homophone(IAugment):
    def __init__(self, dataload: DataLoad):
        super().__init__(dataload=dataload)
        self.config = None
        self.model = None
        self.load_model()

        # save default values of dataset variables
        self.default_char2idx, self.default_idx2char, self.default_max_length = (
            self.dataset.char2idx,
            self.dataset.idx2char,
            self.dataset.max_length,
        )

    def load_model(self):
        model_path = Path(__file__).parents[0] / "trained_models/v1/model.pt"

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint["config"]

            self.model = Transformer(
                checkpoint["INPUT_DIM"],
                checkpoint["OUTPUT_DIM"],
                checkpoint["config"]["max_length"],  # max word length
                checkpoint["config"]["max_output_len"],
                checkpoint["HEADS"],
                checkpoint["LAYERS"],
                checkpoint["LAYERS"],
                checkpoint["HID_DIM"],
                0,
            ).to(self.device)

            self.model = torch.nn.DataParallel(self.model)

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint,
                strict=False,
            )
            logging.info("Model successfully loaded..")

        else:
            raise FileNotFoundError(
                "please make sure model weights are available in model path.."
            )

        self.model.eval()

    def batch_compute(
        self,
        queries: np.array,
        query_tensors: torch.Tensor,
        beam_width: int = 3,
        count_per_query: int = 10,
    ) -> List[tuple]:
        """get homophones for a batch array of strings (VECTORIZED BEAM SEARCH)

        Args:
            queries (List[str]): array of queries
            query_tensors (torch.Tensor): padded query tensors. shape - [# queries, max_sent_len]
            beam_width (int, optional): number of options to consider at each character prediction. Defaults to 3.
            count_per_query (int, optional): result count per query. Defaults to 10.

        Returns:
            List[tuple]: list of (homophone, query)
        """
        self.model.eval()

        unique_queries = set(queries)
        number_of_queries = len(queries)

        # output count for the overall result
        count = count_per_query * number_of_queries

        # shape decoder_inp --> [# queries, 1]
        # for each query, initialize decoder input with <SOS>
        decoder_inp = torch.full(
            (number_of_queries, 1), self.config["char2idx"]["<SOS>"], device=self.device
        )

        # shape output_score --> [# queries]
        # for each query, initialize output score with 1
        output_score = torch.full((number_of_queries,), 1, device=self.device)

        # iterate over max_output_len to generate top n characters (n=beam_width)
        for i in range(1, self.config["max_output_len"]):
            src_mask = (query_tensors != 0).unsqueeze(1)

            trg_tensor = decoder_inp.to(self.device)
            trg_mask = (trg_tensor != 0).unsqueeze(-2)
            dimension = trg_tensor.size(-1)

            mask = torch.tril(torch.ones(1, dimension, dimension, device=self.device))
            mask[mask != 0] = 1
            mask = Variable(mask > 0)

            if trg_tensor.is_cuda:
                mask.cuda()
            trg_mask = trg_mask & mask

            with torch.no_grad():
                output = self.model.forward(
                    query_tensors, trg_tensor, src_mask, trg_mask
                )[:, i - 1, :].unsqueeze(1)

                # [min(# queries * beam_width, count), 1, len(char2idx)]
                output = torch.nn.Softmax(dim=-1)(output)

                # [min(# queries * beam_width, count), len(char2idx)]
                output = output.squeeze(1)

            # sort output and get top n idx where n = beam_width
            values, topk = torch.sort(output, dim=-1, descending=True)

            # [min(# queries * beam_width, count), beam_width]
            values, topk = values[:, :beam_width], topk[:, :beam_width]

            # update the output score
            output_score = (
                (values * output_score.unsqueeze(-1)).transpose(1, 0).reshape(-1)
            )

            # repeat decoder_inp beam_width times, shape - [min(# queries * beam_width, count) * beam_width, i]
            # concat the top n characters to the decoder_inp
            # shape decoder_inp - [min(# queries * beam_width, count) * beam_width, i + 1]
            decoder_inp = torch.cat(
                (
                    decoder_inp.repeat(beam_width, 1),
                    topk.transpose(1, 0).reshape(-1).unsqueeze(-1),
                ),
                -1,
            )

            # repeat query and queries beam_width times
            # shape query - [min(# queries * beam_width, count) * beam_width, max_sent_len]
            # shape queries - [min(# queries * beam_width, count)]
            query_tensors = query_tensors.repeat(beam_width, 1)
            queries = queries.reshape(1, -1).repeat(beam_width, 0).reshape(-1)

            top_count_indices = []

            # selecting count_per_query outputs for each unique query
            for query_ in unique_queries:
                query_idx = (queries == query_).nonzero()[0]
                top_count_indices.extend(
                    query_idx[
                        torch.argsort(output_score[query_idx], descending=True)[
                            :count_per_query
                        ].cpu()
                    ]
                )

            # index decoder_inp, query, output_score and queries using top_count_indices
            decoder_inp = decoder_inp[top_count_indices]
            query_tensors = query_tensors[top_count_indices]
            output_score = output_score[top_count_indices]
            queries = queries[top_count_indices]

            # terminate the loop if <EOS> is encounetered atleast once for each unique query
            if (decoder_inp == self.config["char2idx"]["<EOS>"]).any(-1).all():
                break

        # iterate on range of min(decoder_inp.shape[0], count) to convert idx to characters
        # and join on the specified output separator
        out = [
            (
                "".join([self.config["idx2char"][j.item()] for j in decoder_inp[i]][1:])
                .split("<EOS>", 1)[0]
                .strip(),
                queries[i],
            )
            for i in range(min(decoder_inp.shape[0], count))
            if len(queries[i]) > 3 and not REGEXP.search(queries[i])
        ]

        # return unique outputs
        return list(dict.fromkeys(out))

    def compute(
        self, beam_width: int, count_per_query: int, progress_bar: bool = True
    ) -> List[tuple]:
        """Get homophones for a list of words

        Args:
            beam_width (int): number of options to consider at each character prediction
            count_per_query (int): result count per query

        Returns:
            List[tuple]: list of (homophone, query)
        """
        # set the variables of the default dataset object as per the homophones module
        self.dataset.char2idx, self.dataset.idx2char, self.dataset.max_length = (
            self.config["char2idx"],
            self.config["idx2char"],
            self.config["max_length"],
        )

        pbar = tqdm(self.dataloader, disable=not progress_bar)
        pbar.set_description(
            f"Homophones: count_per_query {count_per_query}, beam_width {beam_width}"
        )

        output = set()

        for _, (tensor, queries) in enumerate(pbar, 0):
            tensor = tensor.to(self.device)
            queries = np.array(queries)

            output.update(
                self.batch_compute(queries, tensor.clone(), beam_width, count_per_query)
            )

        # reset the variables back to the default values
        self.dataset.char2idx, self.dataset.idx2char, self.dataset.max_length = (
            self.default_char2idx,
            self.default_idx2char,
            self.default_max_length,
        )

        torch.cuda.empty_cache()
        return list(output)

    def __call__(
        self, beam_width: int, count_per_query: int, progress_bar: bool = True
    ):
        """Get homophones for a list of words

        Args:
            beam_width (int): number of options to consider at each character prediction
            count_per_query (int): result count per query

        Returns:
            List[tuple]: list of (homophone, query)
        """
        return self.compute(
            beam_width=beam_width,
            count_per_query=count_per_query,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    from nla.augment import Augment

    queries = ["CROCIN", "TABLET", "PARACETAMOL", "IVERMECTIN", "CIPLA"]

    aug = Augment(queries=queries, batch_size=3)
    homophone = aug(aug="homophone")

    beam_width = 3
    count_per_query = 10
    print(homophone(beam_width=beam_width, count_per_query=count_per_query))

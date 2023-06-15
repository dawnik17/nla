from typing import List

import torch
from typing import Union
import numpy as np
from nla.augment.device import get_torch_device


class TypoEngine:
    def __init__(self):
        self.device = get_torch_device()

    def insert1D(
        self, tensor: torch.Tensor, idx: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        value = torch.tensor(-1), idx = torch.tensor(4);

        tensor([1, 2, 3, 0, 0, 0, 0])----------->from this
                                                        |
        tensor([ 1,  2,  3,  0, -1,  0,  0,  0])<--to this

        Args:
            tensor (torch.Tensor): base tensor (1D)
            idx (torch.Tensor): index before which we want to add the value
            value (torch.Tensor): value that needs top be added

        Returns:
            torch.Tensor: _description_
        """
        return torch.cat(
            [tensor[:idx], value.unsqueeze(-1), tensor[idx:]],
            dim=-1,
        )

    def swap1D(
        self, tensor: torch.Tensor, idx_x: torch.Tensor, idx_y: torch.Tensor
    ) -> torch.Tensor:
        """
        idx = torch.tensor(1);

        tensor([1, 2, 3, 0, 0, 0, 0])----------->from this
                                                        |
        tensor([1, 3, 2, 0, 0, 0, 0])<--to this---------

        Args:
            tensor (torch.Tensor): base tensor (1D)
            idx_x (torch.Tensor): index which we want to swap with idx_y
            idx_y (torch.Tensor): index which we want to swap with idx_x

        Returns:
            torch.Tensor: _description_
        """
        tensor[[idx_x, idx_y]] = tensor[[idx_y, idx_x]]
        return tensor

    def replace1D(
        self, tensor: torch.Tensor, idx: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        value = torch.tensor(-1), idx = torch.tensor(2);

        tensor([1, 2, 3, 0, 0, 0, 0])----------->from this
                                                        |
        tensor([1, 2, -1, 0, 0, 0, 0])<--to this--------

        Args:
            tensor (torch.Tensor): base tensor (1D)
            idx (torch.Tensor): index before which we want to add the value
            value (torch.Tensor): value that needs top be added

        Returns:
            torch.Tensor: _description_
        """
        tensor[idx] = value
        return tensor

    def delete1D(self, tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        idx = torch.tensor(2);

        tensor([1, 2, 3, 0, 0, 0, 0])----------->from this
                                                        |
        tensor([1, 2, 0, 0, 0, 0])<--to this------------

        Args:
            tensor (torch.Tensor): base tensor (1D)
            idx (torch.Tensor): index which we want to remove
            value (int): value that needs top be added

        Returns:
            torch.Tensor: _description_
        """
        return torch.cat(
            [tensor[:idx], tensor[idx + 1 :]],
            dim=-1,
        )

    def replace2D(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        indices = torch.tensor([1, 2, 3]), values = torch.tensor([-1, -2, -3])

        tensor([[1, 2, 3, 4],
                [1, 2, 3, 4],  ---------from this--
                [1, 2, 3, 4]])                     |
                                                   |
        tensor([[ 1, -1,  3,  4],                  |
                [ 1,  2, -2,  4],<-------to this---
                [ 1,  2,  3, -3]])

        Args:
            tensor (torch.Tensor): _description_
            indices (List[int]): _description_
            values (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        tensor[torch.arange(tensor.size(0)), indices] = values
        return tensor

    def delete2D(
        self, tensor: torch.Tensor, indices: torch.Tensor, empty_space_id: torch.Tensor
    ) -> torch.Tensor:
        """indices = torch.tensor([0, 1, 2]), empty_space_id = torch.tensor(0)

        tensor([[1, 2, 3, 4, 0, 0],
                [1, 2, 3, 4, 0, 0], ----->from this--
                [1, 2, 3, 4, 0, 0]])                 |
                                                     |
                                                     |
        tensor([[0, 2, 3, 4, 0, 0],                  |
                [1, 0, 3, 4, 0, 0], <----to this-----
                [1, 2, 0, 4, 0, 0]])

        While converting this tensor to string,
        the empty_space_ids(0s) will get deleted since they are empty spaces.

        Args:
            tensor (torch.Tensor): _description_
            indices (List[int]): _description_
            empty_space_id (int, optional): _description_. Defaults to 0.

        Returns:
            torch.Tensor: _description_
        """
        return self.replace2D(tensor=tensor, indices=indices, values=empty_space_id)

    def swap2D(
        self, tensor: torch.Tensor, indices_x: torch.Tensor, indices_y: torch.Tensor
    ) -> torch.Tensor:
        """
        indices_x = torch.tensor([1,2,3]), indices_y = torch.tensor([2,3,4])

        tensor([[1, 2, 3, 4, 0, 0],
                [1, 2, 3, 4, 0, 0], --> from this ---
                [1, 2, 3, 4, 0, 0]])                 |
                                                     |
        tensor([[1, 3, 2, 4, 0, 0],                  |
                [1, 2, 4, 3, 0, 0], <---to this------
                [1, 2, 3, 0, 4, 0]])

        Args:
            tensor (torch.Tensor): _description_
            indices (List[int]): _description_

        Returns:
            torch.Tensor: _description_
        """
        index_mask_x = tensor == torch.gather(
            tensor, dim=1, index=indices_x.unsqueeze(-1)
        )
        value_x = tensor[index_mask_x]

        index_mask_y = tensor == torch.gather(
            tensor, dim=1, index=indices_y.unsqueeze(-1)
        )
        value_y = tensor[index_mask_y]

        tensor[torch.arange(tensor.size(0)), indices_x] = value_y
        tensor[torch.arange(tensor.size(0)), indices_y] = value_x
        return tensor

    def insert2D(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        indices = torch.tensor([1,2,3]), values = torch.tensor([-1, -2, -3])

        tensor([[1, 2, 3, 4, 0, 0],
                [1, 2, 3, 4, 0, 0], -----> from this ---
                [1, 2, 3, 4, 0, 0]])                    |
                                                        |
        tensor([[ 1, -1,  2,  3,  4,  0,  0],           |
                [ 1,  2, -2,  3,  4,  0,  0], <---to this
                [ 1,  2,  3, -3,  4,  0,  0]])

        Args:
            tensor (torch.Tensor): _description_
            indices (List[int]): _description_
            values (List[int]): _description_

        Returns:
            torch.Tensor: _description_
        """
        # indices = np.add(indices, np.random.choice([0, 1], size=len(indices)))

        output = []

        for i in range(tensor.size(0)):
            output.append(self.insert1D(tensor[i], indices[i], values[i]))

        return torch.stack(output)

    def replace3D(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        indices = torch.tensor([[1,2,3],[0,1,2]]),
        values = torch.tensor([[-1,-2,-3],[-3,-1,-2]])

        tensor([[[0.5888, 0.6391, 0.4958, 0.8019],
                 [0.2475, 0.5830, 0.4571, 0.3393],
                 [0.4214, 0.8748, 0.4808, 0.0797]],
                                                    ---from this--->
                [[0.1770, 0.5666, 0.0623, 0.7545],                  |
                 [0.2157, 0.4041, 0.5920, 0.9543],                  |
                 [0.9508, 0.7266, 0.8462, 0.3252]]])                |
                                                                    |
        tensor([[[ 0.5888, -1.0000,  0.4958,  0.8019],              |
                 [ 0.2475,  0.5830, -2.0000,  0.3393],              |
                 [ 0.4214,  0.8748,  0.4808, -3.0000]],             |
                                                       <---to this--
                [[-3.0000,  0.5666,  0.0623,  0.7545],
                 [ 0.2157, -1.0000,  0.5920,  0.9543],
                 [ 0.9508,  0.7266, -2.0000,  0.3252]]])

        Args:
            tensor (torch.Tensor): _description_
            indices (torch.Tensor): _description_
            values (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return tensor.scatter(
            dim=-1, index=indices.unsqueeze(-1), src=values.unsqueeze(-1)
        )

    def delete3D(
        self, tensor: torch.Tensor, indices: torch.Tensor, empty_space_id: torch.Tensor
    ) -> torch.Tensor:
        """
        indices = torch.tensor([[1,2,3],[0,1,2]]),
        empty_space_id = torch.tensor(0)

        tensor([[[0.5888, 0.6391, 0.4958, 0.8019],
                 [0.2475, 0.5830, 0.4571, 0.3393],
                 [0.4214, 0.8748, 0.4808, 0.0797]],
                                                    ---from this--->
                [[0.1770, 0.5666, 0.0623, 0.7545],                  |
                 [0.2157, 0.4041, 0.5920, 0.9543],                  |
                 [0.9508, 0.7266, 0.8462, 0.3252]]])                |
                                                                    |
        tensor([[[ 0.5888,  0.0000,  0.4958,  0.8019],              |
                 [ 0.2475,  0.5830,  0.0000,  0.3393],              |
                 [ 0.4214,  0.8748,  0.4808,  0.0000]],             |
                                                       <---to this--
                [[ 0.0000,  0.5666,  0.0623,  0.7545],
                 [ 0.2157,  0.0000,  0.5920,  0.9543],
                 [ 0.9508,  0.7266,  0.0000,  0.3252]]])

        Args:
            tensor (torch.Tensor): _description_
            indices (torch.Tensor): _description_
            values (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        empty_space_id = empty_space_id.repeat(indices.size())
        return self.replace3D(tensor=tensor, indices=indices, values=empty_space_id)

    def swap3D(
        self, tensor: torch.Tensor, indices_x: torch.Tensor, indices_y: torch.Tensor
    ) -> torch.Tensor:
        """
        indices_x = torch.tensor([[1,2,2],[0,1,2]]), indices_y = torch.tensor([[2,3,3],[1,2,3]])

        tensor([[[0.0467, 0.5789, 0.6343, 0.9608],
                [0.1462, 0.9436, 0.9451, 0.5046],
                [0.1366, 0.7674, 0.8585, 0.3093]],
                                                    ---->from this--
                [[0.6568, 0.7023, 0.4334, 0.8467],                  |
                [0.2452, 0.8304, 0.4980, 0.3470],                   |
                [0.2067, 0.5879, 0.1785, 0.4536]]])                 |
                                                                    |
        tensor([[[0.0467, 0.6343, 0.5789, 0.9608],                  |
                [0.1462, 0.9436, 0.5046, 0.9451],                   |
                [0.1366, 0.7674, 0.3093, 0.8585]],                  |
                                                    <--to this------
                [[0.7023, 0.6568, 0.4334, 0.8467],
                [0.2452, 0.4980, 0.8304, 0.3470],
                [0.2067, 0.5879, 0.4536, 0.1785]]])

        Args:
            tensor (torch.Tensor): _description_
            indices_x (torch.Tensor): _description_
            indices_y (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        size0 = tensor.size(0)
        size1 = tensor.size(1)

        axis_x = torch.arange(size0).repeat_interleave(size1)
        axis_y = torch.arange(size1).repeat(size0)

        indices_x = indices_x.flatten()
        indices_y = indices_y.flatten()

        temp = tensor[axis_x, axis_y, indices_x]
        tensor[axis_x, axis_y, indices_x] = tensor[axis_x, axis_y, indices_y]
        tensor[axis_x, axis_y, indices_y] = temp

        return tensor

    def insert3D(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        indices = torch.tensor([[1, 2, 2], [0, 1, 2]])
        values = torch.tensor([[-1, -2, -3], [-3, -1, -2]])
        tensor([[[0.0467, 0.5789, 0.6343, 0.9608],
                [0.1462, 0.9436, 0.9451, 0.5046],
                [0.1366, 0.7674, 0.8585, 0.3093]],
                                                    ----from this---------->
                [[0.6568, 0.7023, 0.4334, 0.8467],                          |
                [0.2452, 0.8304, 0.4980, 0.3470],                           |
                [0.2067, 0.5879, 0.1785, 0.4536]]])                         |
                                                                            |
                                                                            |
        tensor([[[ 0.0467, -1.0000,  0.5789,  0.6343,  0.9608],             |
                 [ 0.1462,  0.9436, -2.0000,  0.9451,  0.5046],             |
                 [ 0.1366,  0.7674, -3.0000,  0.8585,  0.3093]],            |
                                                                <-to this---
                [[-3.0000,  0.6568,  0.7023,  0.4334,  0.8467],
                 [ 0.2452, -1.0000,  0.8304,  0.4980,  0.3470],
                 [ 0.2067,  0.5879, -2.0000,  0.1785,  0.4536]]])
        Args:
            tensor (torch.Tensor): _description_
            indices (torch.Tensor): _description_
            values (torch.Tensor): _description_
        Returns:
            torch.Tensor: _description_
        """
        output = []

        for i in range(tensor.size(0)):
            output.append(self.insert2D(tensor[i], indices[i], values[i]))

        return torch.stack(output)

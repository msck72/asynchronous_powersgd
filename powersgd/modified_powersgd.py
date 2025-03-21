from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import allreduce_average, pack, unpack, is_distributed

ALPHA = 0.1
# How much is too much??
EFFECT = 0.1

class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor], group, dist_group_id, timer) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass


class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor], group, dist_group_id, timer) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        with timer('only_compress'):
            buffer, shapes = pack(gradients)
        with timer('only_all_reduce'):
            allreduce_average(buffer, group=group)
        with timer('only_decompress'):
            out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor], dist_group, dist_group_id, timer) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate(gradients, dist_group, dist_group_id, timer)

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate(compressed_grads, dist_group, dist_group_id, timer),
            self._allreduce.aggregate(uncompressed_grads, dist_group, dist_group_id, timer),
        )

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

        # remember the history gradient
        self.history_gradient = None
        # remember the previous round group that I belonged to
        self.prev_round_group = torch.distributed.get_rank()
        
        

    def aggregate(self, gradients: List[torch.Tensor], dist_group, dist_group_id, timer) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """
        if self.history_gradient is None:
            self.history_gradient = {shape:torch.stack(hg_same_shape) for shape, hg_same_shape in self._matrices_per_shape([torch.zeros_like(g) for g in gradients]).items()}
            
            
        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]
        
        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        
        # add the history in case it is present    
        
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        
        
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices) + self.history_gradient[shape],
                history=self.history_gradient[shape],
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]



        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])), 
                    in_batch, 
                    out=out_batch
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )

            # Average across workers
            if is_distributed():
                num_workers = torch.distributed.get_world_size(dist_group)

                # In this particular round, get the details of all the participating nodes immediate history(the group in which they have participated in last round)
                prev_rnd_groups = [torch.tensor([0]) for _ in range(num_workers)]
                torch.distributed.all_gather(prev_rnd_groups, torch.tensor([self.prev_round_group]), dist_group)

                prev_rnd_groups = torch.stack(prev_rnd_groups, dim=1).unsqueeze(0)
                # calculate how many belonged to the same group as this in previous round
                num_same_prev_grp = (prev_rnd_groups == self.prev_round_group).sum().item()

                # calculate how many are unique
                eff_num_workers = len(torch.unique(prev_rnd_groups))

                self.prev_round_group = dist_group_id

                # multipy p and q with their respective prev same group size participating in this round
                torch.distributed.all_reduce(self._ps_buffer, group=dist_group)
                torch.distributed.all_reduce(self._qs_buffer, group=dist_group)

                # self._ps_buffer.mul_((1 / num_workers))
                # self._qs_buffer.mul_((1 / num_workers))

            else:
                num_workers = 1

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    # alpha=1 / num_workers
                )
        
        # remove the history that you have seen    
        for group in shape_groups:
            group['approximation'].sub_(group['history'])
            group['approximation'].mul_((1 / (num_workers * num_workers)))
        
        # save this history to the history
        for group in shape_groups:
            group['history'].copy_(group['approximation'])
        
        
        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Increment the step counter
        self.step_counter += 1
        
        return output_tensors
        
    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats



def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)

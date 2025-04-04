"""
In this asynchronous parameters synchgronization in the synchronization step we estimate the change of parameters instead of estimating the parameters.
WITH SCALING AND FEEDBACK in PARAMETERS
"""



from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union
import torch.distributed as dist

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import allreduce_average, pack, unpack, is_distributed

import json
import sys


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor], group, dist_group_id, timer) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass
    
    @abstractmethod
    def update_low_rank_weights(self, parameters: List[torch.Tensor], timer) -> None:
        pass
    @abstractmethod
    def aggregate_parameters(self, parameters: List[torch.Tensor], timer) -> List[torch.Tensor]:
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

    def update_low_rank_weights(self, parameters: List[torch.Tensor], timer) -> None:
        return
    
    def aggregate_parameters(self, parameters: List[torch.Tensor], timer) -> List[torch.Tensor]:
        if len(parameters) == 0:
            return []
        with timer('only_compress'):
            buffer, shapes = pack(parameters)
        with timer('only_all_reduce'):
            allreduce_average(buffer, group=dist.group.WORLD)
        with timer('only_decompress'):
            out = unpack(buffer, shapes)
        
        # not required as the we are not using error feedback for parameters 
        # for p in parameters:
        #     p.zero_()
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
        
    def update_low_rank_weights(self, parameters: List[torch.Tensor], timer) -> None:
        compressed_params, _ = self._split(parameters)
        self._powersgd.update_low_rank_weights(compressed_params, timer)
        return
    
    def aggregate_parameters(self, parameters: List[torch.Tensor], timer) -> List[torch.Tensor]:
        
        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate_parameters(parameters, timer)

        compressed_params, uncompressed_params = self._split(parameters)
        return self._merge(
            self._powersgd.aggregate_parameters(compressed_params, timer),
            self._allreduce.aggregate_parameters(uncompressed_params, timer),
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
                self._init_p_batch(shape, params, self.config.rank)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params, self.config.rank)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)
        
        # related to parameters
        
        self.synchronization_freq = 20
        self.curr_round = 0
        # self.saved = False
        
        # save the mode parameters initially
        self.model_parameters_track: List[torch.Tensor] = [p.clone().detach() for p in params]
        self.model_parameters_per_shape = self._matrices_per_shape(self.model_parameters_track)
        
        
        self._ps_params_buffer, ps_params_shapes = pack(
            [
                self._init_p_batch(shape, params, self.config.rank)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps_params = unpack(self._ps_params_buffer, ps_params_shapes)

        self._qs_params_buffer, qs_params_shapes = pack(
            [
                self._init_q_batch(shape, params, self.config.rank)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs_params = unpack(self._qs_params_buffer, qs_params_shapes)

    def aggregate(self, gradients: List[torch.Tensor], dist_group, dist_group_id, timer) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """  
        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]
        
        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
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
                out_buffer, other_buffer = self._qs_buffer, self._ps_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer, other_buffer = self._ps_buffer, self._qs_buffer
            
            if is_distributed():
                # if torch.get_rank == dist_group_id:
                # then make other_buffers 0s
                # allreduce to get the effective in_buffers
                if torch.distributed.get_rank() != dist_group_id:
                    other_buffer.zero_()
                torch.distributed.all_reduce(other_buffer, group=dist_group)

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
                torch.distributed.all_reduce(out_buffer, group=dist_group)

            else:
                num_workers = 1

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    alpha=1 / num_workers
                )

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
    
    
    
    def update_low_rank_weights(self, parameters: List[torch.Tensor], timer=None) -> None:
        
        # get the difference to estimate
        if self.curr_round == self.synchronization_freq - 1:
            self.curr_round = 0
        
        # if torch.distributed.get_rank() == 0:
        #     tensor_lists = [p.numpy().tolist() for p in parameters]
        #     with open(f"something_{self.curr_round}_before.txt", "w") as f:
        #         f.write(str(tensor_lists))
                
        for p, mp in zip(parameters, self.model_parameters_track):
            p.mul_((self.synchronization_freq - self.curr_round))
            p.sub_(mp)
        
        # if torch.distributed.get_rank() == 0:
        #     tensor_lists = [p.numpy().tolist() for p in parameters]
        #     with open(f"something_{self.curr_round}_after.txt", "w") as f:
        #         f.write(str(tensor_lists))
        
        self.curr_round += 1
        
        # estimate the difference using low-rank matrices
        
        parameters_per_shape = self._matrices_per_shape(parameters)
        
        shape_groups = [
            dict(
                shape=shape,
                params=matrices,
                param_batch=torch.stack(matrices),
                model_params_track_shape=self.model_parameters_per_shape[shape],
            )
            for shape, matrices in list(parameters_per_shape.items())
        ]
        
 
        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs_params, self._ps_params
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps_params, self._qs_params

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["param_batch"])), 
                    in_batch, 
                    out=out_batch
                )
            
            # calculate the error-feedback to be given
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["param_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )
        
        for group in shape_groups:
            for error_gi, model_parameters_gi in zip(
                group["param_batch"], group["model_params_track_shape"]
            ):
                model_parameters_gi.add_(error_gi)
        
        return
    
    def aggregate_parameters(self, parameters: List[torch.Tensor], timer=None) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average parameters by communicating with other workers.
        """
        
        # no compression synchronization 
        no_compression_weights = []
        total_num_workers = dist.get_world_size()
        for p in parameters:
            tensor_copy = p.clone().detach()
            dist.all_reduce(tensor_copy)
            no_compression_weights.append(tensor_copy.mul_(1 / total_num_workers))
            
        torch.save(no_compression_weights, 'no_compression_weights.pth')
        
        output_tensors = [torch.empty_like(p) for p in parameters]

        # get the difference to estimate
        for p, mp in zip(parameters, self.model_parameters_track):
            p.sub_(mp)
        
        parameters_per_shape = self._matrices_per_shape(parameters)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        
        prev_model_parameters = self._matrices_per_shape(self.model_parameters_track)
        
        shape_groups = [
            dict(
                shape=shape,
                params=matrices,
                outputs=outputs_per_shape[shape],
                param_batch=torch.stack(matrices),
                prev_model_parameters=prev_model_parameters[shape],
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(parameters_per_shape.items())
        ]
        
 
        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs_params, self._ps_params
                out_buffer, other_buffer = self._qs_params_buffer, self._ps_params_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps_params, self._qs_params
                out_buffer, other_buffer = self._ps_params_buffer, self._qs_params_buffer
                
                
            if is_distributed():
                # if torch.get_rank == dist_group_id:
                # then make other_buffers 0s
                # allreduce to get the effective in_buffers
                if torch.distributed.get_rank() != 0:
                    other_buffer.zero_()
                torch.distributed.all_reduce(other_buffer)


            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                # print(f'group[param_batch] = {group["param_batch"].shape} in_batch = {in_batch.shape} out_batch = {out_batch.shape}')
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["param_batch"])), 
                    in_batch, 
                    out=out_batch
                )

            # This part is not required as we donot use the error feedback in parameters
            # for group, in_batch, out_batch in zip(
            #     shape_groups, in_batches, out_batches
            # ):
            #     maybe_transpose(group["param_batch"]).baddbmm_(
            #         in_batch, 
            #         batch_transpose(out_batch), 
            #         alpha=-1
            #     )
   
            if is_distributed():
                num_workers = torch.distributed.get_world_size()
                torch.distributed.all_reduce(out_buffer)
                out_buffer.mul_(1 / num_workers)
            else:
                num_workers = 1
    
            
            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                )

        # add the new approximation to the prev_model_parameters and set it to the output
        for group in shape_groups:
            for o, pmp, approx in zip(
                group["outputs"], group["prev_model_parameters"], group["approximation"]
            ):
                # if self.step_counter > 100 and not self.saved:
                #     print(approx)
                o.copy_(pmp.add_(approx))
        
        # print(self.step_counter)
        # if self.step_counter > 100 and not self.saved:
            # ot_human_readable = [ot.tolist() for ot in output_tensors]
            # with open(f'model_{torch.distributed.get_rank()}.json', 'w') as f:
            #     json.dump(ot_human_readable, f, indent = 4)
            # self.saved = True
            # print(output_tensors)
            # print("Saved successfully")
            
            
            
        # estimated sync_weights is output_tensors
        # now we need to get the actual without compression synchronized weights and print 
        # torch.save(output_tensors, 'compressed_weights.pth')
        # sys.exit()

        return output_tensors
    

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor], config_rank: int, 
    ) -> torch.Tensor:
        rank = min(config_rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor], config_rank: int,
    ) -> torch.Tensor:
        rank = min(config_rank, min(shape))
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

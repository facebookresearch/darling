import torch
from megatron.core import parallel_state as mpu

from .sequence_parallel import pad_to_sequence_parallel


def compute_transformers_input_shapes(batches, meta_info):
    from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron

    # pre-compute input shapes for each micro-batch at each pp stage
    input_shapes = []
    for model_inputs in batches:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        if meta_info["sequence_parallel"]:
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # compute shapes for model_inputs
            input_shapes.append(
                torch.Size(
                    [
                        input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(),
                        1,
                        meta_info["hidden_size"],
                    ]
                )
            )
        else:
            # compute shapes for model_inputs
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info["hidden_size"]]))
    return input_shapes


def make_batch_generator(batches, vpp_size):
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator

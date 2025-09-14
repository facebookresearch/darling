from typing import List, Optional

from transformers import PreTrainedTokenizer
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizers import *
from vllm.utils import LRUCache


class TokenizerGroup:
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer: PreTrainedTokenizer, enable_lora: bool, max_num_seqs: int, max_input_length: Optional[int]):
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = tokenizer
        self.lora_tokenizers = LRUCache[PreTrainedTokenizer](capacity=max_num_seqs) if enable_lora else None

    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        return True

    def get_max_input_len(self, lora_request: Optional[LoRARequest] = None) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self.max_input_length

    def encode(self, prompt: str, request_id: Optional[str] = None, lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = self.get_lora_tokenizer(lora_request)
        return tokenizer.encode(prompt)

    async def encode_async(self, prompt: str, request_id: Optional[str] = None, lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        return tokenizer.encode(prompt)

    def get_lora_tokenizer(self, lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            # TODO(sgm): the lora tokenizer is also passed, but may be different
            tokenizer = self.tokenizer
            # tokenizer = (get_lora_tokenizer(
            #     lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)

    # FIXME(sgm): for simplicity, we assign the special token here
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

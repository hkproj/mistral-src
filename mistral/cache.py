import torch
from typing import List, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias # Mask for the attention
    seqlens: List[int] 


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor: # seqlen is the total number of tokens cached so far, including the overwritten one. This is needed to calculate the rotation point of the cache
    assert cache.ndim == 3  # (Sliding_Window_Size, Num_Heads, Head_Dim)
    position = seqlen % cache.shape[0] # This is the pivot point around which we need to rotate the cache
    if seqlen < cache.shape[0]: # If the total sequence length so far is smaller than the cache size, then just return the first seqlen elements, as the cache didn't have any rotations yet
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0) # Select the unrotated elements from the cache around the pivot point


class CacheView:
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim) # (Max_Batch_Size, Sliding_Window_Size, N_Heads_KV, Head_Dim) --> (Max_Batch_Size * Sliding_Window_Size, N_Heads_KV, Head_Dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim) # (Max_Batch_Size, Sliding_Window_Size, N_Heads_KV, Head_Dim) --> (Max_Batch_Size * Sliding_Window_Size, N_Heads_KV, Head_Dim)
        # Copies from the xk and xv tensors to the cache tensors, based on the cache positions and the items to cache (to_cache_mask)
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3 # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(Seq, N_Heads_KV, Head_Dim)]
        xk = torch.split(xk, self.metadata.seqlens) # (Seq1+Seq2+Seq3, N_Heads_KV, Head_Dim) --> [(Seq1, N_Heads_KV, Head_Dim), (Seq2, N_Heads_KV, Head_Dim), (Seq3, N_Heads_KV, Head_Dim)]
        xv = torch.split(xv, self.metadata.seqlens) # (Seq1+Seq2+Seq3, N_Heads_KV, Head_Dim) --> [(Seq1, N_Heads_KV, Head_Dim), (Seq2, N_Heads_KV, Head_Dim), (Seq3, N_Heads_KV, Head_Dim)]
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)] # Currently cached elements, already unrotated, one for each prompt
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)] # Currently cached elements, already unrotated, one for each prompt

        interleaved_k = interleave_list(cache_k, xk) # Appends the incoming keys and values to the currently cached elements (one for each prompt)
        interleaved_v = interleave_list(cache_v, xv) # Appends the incoming keys and values to the currently cached elements (one for each prompt)

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self): 
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim # model_dim / n_heads

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist() # Indicates the total length seen by the cache so far (including the overwritten elements) for each prompt

        assert len(seqlens) > 0, seqlens

        # [True] if the token position belongs to the last `sliding_window` positions of the sequence. It is always True unless the chunk size is bigger than the sliding window
        # Indicates which items in the sequence should be cached (the last `sliding_window` tokens of each sequence)
        masks = [ 
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens # The sequence length of each input in the batch (so we can understand which token belongs to which prompt)
        ]

        # Indicates which items in the sequence should be cached (the last `sliding_window` tokens of each sequence)
        # Concatenate all the masks of each prompt in the batch
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool) 

        # Number of elements in the mask == True
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)

        # The position of each token in the prompt (all concatenated). It may not start from zero (because for example the first chunk may be 5 tokens and we are now processing the second chunk)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(device=self.device, dtype=torch.long) 

        # The index of the batch to which each token (in the concatenated list) belongs to.
        batch_idx = torch.tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device, dtype=torch.long) 
        
        # Where each token should be placed in the cache (based on the position in the prompt and the batch index)
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window 

        # Indicates if it is the first prefill (only True on the first chunk)
        first_prefill = seqpos[0] == 0
        # Indicates if it is a subsequent prefill (True from second chunk onwards), but False when generating tokens.
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

        if first_prefill: 
            # For first chunk of prompt. It creates an attention mask that is causal for each prompt and also local based on the sliding window size
            # https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask + local attention based on the sliding window
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window) 
        elif subsequent_prefill:
            # For subsequent chunks of prompt
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, # Size of the query
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)] # The total number of keys and values will be the incoming sequence length + the cached elements 
            ).make_local_attention_from_bottomright(self.sliding_window)
        else: # For token generation
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens, # Size of the query
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist() # The total number of keys and values will be the incoming sequence length + the cached elements 
            )

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )

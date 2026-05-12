import triton
import torch
import triton.language as tl 

from torch import Tensor as T

@triton.jit
def __online_softmax_kernel__(
    output_ptr,
    input_ptr,
    stride_row,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE:  tl.constexpr,
):
    row_idx   = tl.program_id(0)
    row_start = input_ptr + row_idx * stride_row

    # ── PASS 1: online max + sum in ONE sweep ──────
    m = float("-inf")   # running max
    d = 0.0             # running denominator

    for tile_start in range(0, num_cols, BLOCK_SIZE):
        # tile_start is a scalar like "0"
        col_offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < num_cols

        tile = tl.load(row_start + col_offs,
                mask=mask,
                other=float("-inf")).to(tl.float32)

        m_new    = tl.max(tile, axis=0)
        m_new    = tl.maximum(m, m_new)    

        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(tile - m_new), axis=0)
        
        m = m_new
    
    out_start = output_ptr + row_idx * stride_row

    for tile_start in range(0, num_cols, BLOCK_SIZE):
        col_offs    = tile_start + tl.arange(0, BLOCK_SIZE)
        mask        = col_offs < num_cols
        tile        = tl.load(row_start + col_offs,
                              mask=mask,
                              other=float("-inf")).to(tl.float32)
        
        softmax_out = (tl.exp(tile - m) / d).to(OUT_DTYPE)
        tl.store(out_start + col_offs, softmax_out, mask=mask)


def online_softmax(x: T,
                    num_warps: int = 4,
                    block_size: int = 1024) -> T:

    orig_shape = x.shape
    
    x_2d       = x.reshape(-1,  orig_shape[-1])
    total_rows, cols = x_2d.shape
    out        = torch.empty_like(x_2d)
    DTYPE      = tl.float32 if x.dtype == torch.float32 else tl.float16

    num_warps  = 4
    if block_size >= 2048: num_warps = 8
    if block_size >= 4096: num_warps = 16
    grid = (total_rows,)

    __online_softmax_kernel__[grid](
        out,
        x_2d,
        x_2d.stride(0),
        cols,
        BLOCK_SIZE = block_size,
        OUT_DTYPE  = DTYPE,
        num_warps  = num_warps,
    )

    return out.reshape(orig_shape)


if __name__ == "__main__":
    x = torch.rand((3,4,12), dtype=torch.float32, device="cuda")

    triton_out = online_softmax(x,)
    torch_out  = torch.nn.functional.softmax(x, dim=-1) 
    print("Input:\n", x.shape,"\n",x)
    print("\nTriton output:\n", triton_out)
    print("\nInbuild pyTorch output:\n", torch.nn.functional.softmax(x,dim=-1))
    max_diff = (triton_out - torch_out).abs().max().item()
    print(f"\nMax diff: {max_diff:.6f}")
import triton
import triton.language as tl 
import torch

@triton.jit
def __softmax_kernel__(
    output_ptr,    stride_out,
    input_ptr,    stride_input,
    num_cols,
    block_size:tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row_idx          = tl.program_id(0)
    row_start        = input_ptr  + row_idx * stride_input
    col_offsets      = tl.arange(0, block_size)
    input_ptrs       = row_start  + col_offsets
    mask             = col_offsets < num_cols

    row              = tl.load(input_ptrs, mask=mask, other=float("-inf")).to(OUT_DTYPE)
    safe_row         = row - tl.max(row, axis=0)
    exp_row          = tl.exp(safe_row)
    softmax_row      = exp_row / tl.sum(exp_row, axis=0)
    
    softmax_row      = softmax_row.to(OUT_DTYPE)
    out_start        = output_ptr + row_idx * stride_out
    tl.store(out_start + col_offsets, softmax_row, mask=mask)

def triton_softmax_3d(x: torch.Tensor,
                      num_warps: int = 4,
                      block_size: int = None) -> torch.Tensor:
    """Flatten any tensor to (N, cols), run kernel, reshape back."""
    orig_shape = x.shape
    cols       = orig_shape[-1]
    
    x_2d       = x.reshape(-1, cols)
    total_rows, cols = x_2d.shape

    DTYPE = tl.float32 if x.dtype == torch.float32 else tl.float16
    bs         = block_size if block_size else triton.next_power_of_2(cols)
    out        = torch.empty_like(x_2d)
    grid       = (total_rows,)

    __softmax_kernel__[grid](
        out, out.stride(0),
        x_2d,   x_2d.stride(0),
        cols,
        block_size = bs,
        num_warps  = num_warps,
        OUT_DTYPE=DTYPE,
    )
    return out.reshape(orig_shape)

if __name__ == "__main__":
        
    x = torch.rand((3,3,3),device="cuda",dtype=torch.float32)

    print(x.shape,"\n")
    triton_out = triton_softmax_3d(x)
    print("Input:\n", x)
    print("\nTriton output:\n", triton_out)
    print("\nInbuild pyTorch output:\n", torch.nn.functional.softmax(x,dim=-1))
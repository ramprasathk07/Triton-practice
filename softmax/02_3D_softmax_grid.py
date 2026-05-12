import triton
import triton.language as tl 
import torch

@triton.jit
def __softmax_kernel_3d__(
    output_ptr,
    input_ptr,
    stride_batch,    # how many elements to skip per batch
    stride_row,      # how many elements to skip per row
    num_cols,
    block_size: tl.constexpr,
    OUT_DTYPE:  tl.constexpr,
):
    batch_idx        = tl.program_id(0)
    row_idx          = tl.program_id(1)  

    offset      = batch_idx * stride_batch +   row_idx * stride_row
#                  ↑ jumps to correct batch        ↑ jumps to correct row                         

    row_start   = input_ptr  + offset
    out_start   = output_ptr + offset

    col_offsets      = tl.arange(0, block_size)
    input_ptrs       = row_start  + col_offsets
    mask             = col_offsets < num_cols

    row              = tl.load(input_ptrs, mask=mask, other=float("-inf")).to(OUT_DTYPE)
    safe_row         = row - tl.max(row, axis=0)
    exp_row          = tl.exp(safe_row)
    softmax_row      = exp_row / tl.sum(exp_row, axis=0)
    
    softmax_row      = softmax_row.to(OUT_DTYPE)
    tl.store(out_start + col_offsets, softmax_row, mask=mask)

def triton_softmax_3d(x: torch.Tensor,
                      num_warps: int = 4,
                      block_size: int = None) -> torch.Tensor:

    assert x.dim() == 3, "this kernel is 3D only"

    orig_shape = x.shape
    batch, rows , cols = x.shape

    # if x.dim() != 2:
    #     x = x.reshape(-1, x.shape[-1])

    DTYPE = tl.float32 if x.dtype == torch.float32 else tl.float16
    bs         = block_size if block_size else triton.next_power_of_2(cols)
    out        = torch.empty_like(x)
    grid  = (batch, rows)
    num_warps  = 4
    
    if block_size >= 2048: num_warps = 8
    if block_size >= 4096: num_warps = 16
    
    __softmax_kernel_3d__[grid](
        out,
        x,   
        x.stride(0),
        x.stride(1),
        cols,
        block_size = bs,
        num_warps  = num_warps,
        OUT_DTYPE=DTYPE,
    )
    return out.reshape(orig_shape)

if __name__ == "__main__":
        
    x = torch.rand((2,3,6),device="cuda",dtype=torch.float32)

    print(x.shape,"\n")
    triton_out = triton_softmax_3d(x)
    print("Input:\n", x)
    print("\nTriton output:\n", triton_out)
    print("\nInbuild pyTorch output:\n", torch.nn.functional.softmax(x,dim=-1))
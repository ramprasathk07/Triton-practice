import torch

def naive_softmax(x:torch.Tensor, dim=-1)->torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    
    exp_x = torch.exp(x_stable)
    softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return softmax_x

if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0],
                    [1.0, 1.0, 5.1],
                    [1.234, 1.0, 5.1]],device="cuda",dtype=torch.float32)

    print(x.shape,"\n")
    output = naive_softmax(x, dim=-1)
    print("Input:\n", x)
    print("\nSoftmax output:\n", output)
    print("\nInbuild pyTorch output:\n", torch.nn.functional.softmax(x,dim=1))
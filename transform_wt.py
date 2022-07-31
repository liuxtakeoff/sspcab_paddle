import numpy as np
import torch
import paddle
import os
def torch2paddle():
    torch_path = r"C:\Users\Administrator\Desktop\sspcab_paddle\wt_torch.pth"
    paddle_path = r"C:\Users\Administrator\Desktop\sspcab_paddle\wt_paddle.pdparams"
    torch_state_dict = torch.load(torch_path)
    print(torch_state_dict.keys())
    fc_names = ["classifier","out","fc"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    torch2paddle()
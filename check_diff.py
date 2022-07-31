import paddle
import torch
import paddle.nn as nn
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
import numpy as np
from sspcab_torch import SSPCAB as ssp_torch
from tools.model import SSPCAB as ssp_paddle
from tools.model import ProjectionNet as net_paddle
import random
seed = 114514
torch.manual_seed(seed)
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

def create_wt():
    print("======create random weights...=============")
    model_torch = ssp_torch(512)
    wt_torch = model_torch.state_dict()
    torch.save(wt_torch,"wt_torch.pth")
    torch_path = r"wt_torch.pth"
    paddle_path = r"wt_paddle.pdparams"
    torch_state_dict = torch.load(torch_path)
    print(torch_state_dict.keys())
    fc_names = ["classifier", "out", "fc"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:  # ignore bias
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
def check_model():
    """
    检查模型前向转播是否对齐，并返回对齐的模型
    """
    print("======start check model...=============")
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    data_1 = np.random.rand(4,512,7,7).astype(np.float32)
    datap = paddle.to_tensor(data_1)
    datat = torch.tensor(data_1)
    wt_torch = "wt_torch.pth"
    wt_paddle = "wt_paddle.pdparams"
    model_paddle = ssp_paddle(512)
    model_torch = ssp_torch(512)
    model_torch.load_state_dict(torch.load(wt_torch))
    model_paddle.load_dict(paddle.load(wt_paddle))
    model_paddle.eval()
    model_torch.eval()


    datat = model_torch(datat)
    datap = model_paddle(datap)
    reprod_log_1.add("result_model", datap.cpu().detach().numpy())
    reprod_log_1.save("diff_log/result_model_paddle.npy")

    reprod_log_2.add("result_model", datat.cpu().detach().numpy())
    reprod_log_2.save("diff_log/result_model_torch.npy")

    # check_diff
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("diff_log/result_model_paddle.npy")
    info2 = diff_helper.load_info("diff_log/result_model_torch.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_model.txt")
    return model_paddle, model_torch

if __name__ == '__main__':
    # create_wt()
    # model_paddle,model_torch = check_model()

    model_paddle = net_paddle()
    data1 = paddle.rand([32,3,256,256])
    out0,out1,ins,ous = model_paddle(data1)


    print(out0.shape)

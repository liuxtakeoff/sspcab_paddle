
import argparse
import PIL.Image as Image
import pickle
import paddle
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.model import ProjectionNet_sspcab as ProjectionNet
from paddle.vision import transforms
from tools.density import GaussianDensitySklearn,GaussianDensityPaddle
from pathlib import Path

#TODO:集成到一个可以调用的函数里
def predict_img(model,img_path):
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infer img')
    parser.add_argument('--model_dir', default="logs",
                        help=' directory contating models to evaluate (default: models)')
    parser.add_argument('--data_type', default="bottle",
                        help=' type of the input image (default: bottle)')
    parser.add_argument('--img-path', default="./images/demo0.png",
                        help='image size for model infer (default: 256)')
    parser.add_argument('--cuda', default=False,
                        help='use cuda for model predictions (default: False)')
    parser.add_argument('--img_size', default=256,
                        help='image size for model infer (default: 256)')
    parser.add_argument('--dist_th', default=0.5,type=float,
                        help='distance threshold for defect detection (default: 0.5)')
    parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: paddle)')
    args = parser.parse_args()
    print(args)
    if args.density == "sklearn":
        density = GaussianDensitySklearn()
        density.load("%s/%s/kde.crf"%(args.model_dir,args.data_type))
    else:
        density = GaussianDensityPaddle()
        density.load("%s/%s/params.crf" % (args.model_dir, args.data_type))
    with open("%s/%s/minmaxdist.txt"%(args.model_dir,args.data_type),"r") as fd:
        line = fd.readlines()[0].strip().split()
        density.min = float(line[1])
        density.max = float(line[3])
    if ".pdparams" in args.model_dir:
        print("检测到路径包含小数点，判断为直接模型路径，直接读取模型路径！")
        model_name = args.model_dir
    else:
        model_name = "%s/%s/final.pdparams"%(args.model_dir,args.data_type)
    if model_name == None:
        print("warning: cant find the model for %s"%args.data_type)
    print(f"loading model {model_name}")
    head_layers = [512]*1  + [128]
    weights = paddle.load(str(model_name))
    classes = 3
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
    model.set_dict(weights)
    # model.to(device)
    model.eval()

    #定义图像预处理
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((args.img_size, args.img_size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    #加载图片并进行预处理
    img = Image.open(args.img_path)
    img = img.resize((args.img_size,args.img_size)).convert("RGB")
    img = test_transform(img)
    img = img.unsqueeze(0)
    # print(img)

    with paddle.no_grad():
        embed,logit,_,_ = model(img)
        print(embed.shape,logit.shape)
    embed =paddle.nn.functional.normalize(embed, p=2, axis=1)
    distances = density.predict(embed)
    if distances[0] >= args.dist_th:
        print("分类结果为：异常！异常分数为：%.4f"%distances[0])
    else:
        print("分类结果为：正常！异常分数为：%.4f"%distances[0])



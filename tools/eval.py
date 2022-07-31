
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from paddle.vision import transforms
from paddle.io import DataLoader
import paddle
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.dataset import MVTecAT
from tools.model import ProjectionNet_sspcab as ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tools.cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from collections import defaultdict
from tools.density import GaussianDensitySklearn, GaussianDensityPaddle
import pandas as pd
import numpy as np
test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(model, size, defect_type, transform, device="cuda",datadir="Data"):
    # train data / train kde
    test_data = MVTecAT(datadir, defect_type, size, transform=transform, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=32,
                            shuffle=True, num_workers=0)
    train_embed = []
    with paddle.no_grad():
        for x in dataloader_train:
            embed, logit,_,_ = model(x)
            train_embed.append(embed.cpu())
    train_embed = paddle.concat(train_embed)
    return train_embed


def eval_model(modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=False, model=None,
               train_embed=None, head_layer=8, density=GaussianDensityPaddle(),data_dir = "Data"):
    # create test dataset
    global test_data_eval, test_transform, cached_type

    # TODO: cache is only nice during training. do we need it?
    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size, size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT(data_dir, defect_type, size, transform=test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=32,
                                 shuffle=False, num_workers=0)

    # create model
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512] * head_layer + [128]
        weights = paddle.load(str(modelname))
        classes = 3
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.set_dict(weights)
        # model.to(device)
        model.eval()

    # get embeddings for test data
    labels = []
    embeds = []
    with paddle.no_grad():
        for x, label in dataloader_test:
            embed, logit,_,_ = model(x)

            # save
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = paddle.concat(labels)
    embeds = paddle.concat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device,datadir=data_dir)

    # norm embeds
    embeds = paddle.nn.functional.normalize(embeds, p=2, axis=1)
    train_embed = paddle.nn.functional.normalize(train_embed, p=2, axis=1)

    # create eval plot dir
    if save_plots:
        eval_dir = Path("logs") / data_type
        eval_dir.mkdir(parents=True, exist_ok=True)

        # plot tsne
        # also show some of the training data
        show_training_data = False
        if show_training_data:
            # augmentation setting
            # TODO: do all of this in a separate function that we can call in training and evaluation.
            #       very ugly to just copy the code lol
            min_scale = 0.5

            # create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))

            train_transform = transforms.Compose([])
            # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())

            train_data = MVTecAT(data_dir, defect_type, transform=train_transform, size=size)
            dataloader_train = DataLoader(train_data, batch_size=32,
                                          shuffle=True, num_workers=4, collate_fn=cut_paste_collate_fn,
                                          persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with paddle.no_grad():
                for x1, x2 in dataloader_train:
                    x = paddle.concat([x1, x2], axis=0)
                    embed, logit,_,_ = model(x.to(device))

                    # generate labels:
                    y = paddle.to_tensor([0, 1])
                    # y = y.repeat_interleave(x1.size(0))
                    y = paddle.repeat_interleave(y,x1.shape[0])

                    # save
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = paddle.concat(train_labels)
            train_embeds = paddle.concat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = paddle.concat([labels, train_labels + 2])
            tsne_embeds = paddle.concat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")

    print(f"using density estimation {density.__class__.__name__}")
    # density.fit(train_embed,"logs/%s/kde.crf"%defect_type)
    if args.density == "paddle":
        density.fit(train_embed,"logs/%s/params.crf"%defect_type)
    else:
        density.fit(train_embed,"logs/%s/kde.crf"%defect_type)
    distances_train = density.predict(train_embed)
    mind,maxd = min(distances_train),max(distances_train)
    with open("logs/%s/minmaxdist.txt"%data_type,"w") as f_dist:
        f_dist.write("min %.6f max %.6f"%(mind,maxd))
    distances = density.predict(embeds)
    distances = (distances-mind)/(maxd-mind+1e-8)
    # TODO: set threshold on mahalanobis distances and use "real" probabilities
    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    return roc_auc


def plot_roc(labels, scores, filename, modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds.cpu().detach().numpy(), labels.cpu().detach().numpy())
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--model_dir', default="logs",type=str,
                        help=' directory contating models to evaluate (default: logs)')
    parser.add_argument('--data_dir', default="Data",type=str,
                        help=' directory contating datas to evaluate (default: Data)')
    parser.add_argument('--cuda', default=False,
                        help='use cuda for model predictions (default: False)')
    parser.add_argument('--head_layer', default=1, type=int,
                        help='number of layers in the projection head (default: 8)')
    parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: paddle)')
    parser.add_argument('--save_plots', default=False,
                        help='save TSNE and roc plots')
    parser.add_argument('--pretrained', default=None,
                        help='no sense')
    args = parser.parse_args()

    all_types = [
                 'bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper'
                 ]

    if args.type == "all":
        types = all_types
    else:
        types = ["bottle"]
        args.data_dir = "lite_data"
    device = "cuda" if args.cuda in ["True","1","y",True] else "cpu"
    save_plots = True if args.save_plots in ["True","1","y"] else False
    density = GaussianDensitySklearn if args.density == "sklearn" else GaussianDensityPaddle
    print(args)

    # save pandas dataframe
    eval_dir = Path(args.model_dir + "/evalution")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # #找到最佳的训练批次
    # max_aveauroc = 0
    # best_epoch = 0
    # f = open("%s/evalution/epoch_auroc.csv" %args.model_dir, "w")
    # headline = "epoch"
    # for _type in types:
    #     headline+=",%s"%_type
    # headline += ",average\n"
    # f.write("%s,")
    # for epnum in range(100,10000,100):
    #     obj = defaultdict(list)
    #     f.write("%d"%epnum)
    #     for data_type in types:
    #         print(f"evaluating {data_type}")
    #         model_name = "%s/%s/%d.pdparams" % (args.model_dir, data_type,epnum)
    #         roc_auc = eval_model(model_name, data_type, save_plots=save_plots, device=device,
    #                              head_layer=args.head_layer, density=density(), data_dir=args.data_dir)
    #         print(f"{data_type} AUC: {roc_auc}")
    #         obj["defect_type"].append(data_type)
    #         obj["roc_auc"].append(roc_auc)
    #         f.write(",%f"%roc_auc)
    #     ave_auroc = np.mean(obj["roc_auc"])
    #     obj["defect_type"].append("average")
    #     obj["roc_auc"].append(ave_auroc)
    #     f.write(",%f\n"%ave_auroc)
    #     print("epoch%d ave_auroc %.5f"%(epnum,ave_auroc))
    #     if ave_auroc > max_aveauroc:
    #         df = pd.DataFrame(obj)
    #         df.to_csv(str(eval_dir) + "/best_perf.csv")
    #         max_aveauroc = ave_auroc
    #         best_epoch = epnum
    #         print("best epoch: %d max auroc: %.5f"%(epnum,ave_auroc))
    # f.write("best_epoch%d best_ave_auroc %.5f" % (best_epoch, max_aveauroc))
    # f.close()

    obj = defaultdict(list)
    for data_type in types:
        print(f"evaluating {data_type}")
        model_name = "%s/%s/final.pdparams"%(args.model_dir,data_type)
        roc_auc = eval_model(model_name, data_type, save_plots=save_plots, device=device,
                             head_layer=args.head_layer, density=density(),data_dir=args.data_dir)
        print(f"{data_type} AUC: {roc_auc}")
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)
    ave_auroc = np.mean(obj["roc_auc"])
    obj["defect_type"].append("average")
    obj["roc_auc"].append(ave_auroc)
    print("average auroc:%.4f"%ave_auroc)

    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "/total_perf.csv")

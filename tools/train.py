from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import time
import paddle
from paddle import optimizer as optim
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.vision import transforms
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.dataset import MVTecAT, Repeat
from tools.cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from tools.model import ProjectionNet_sspcab as ProjectionNet
from tools.eval import eval_model
import os
import random
import numpy as np
seed = 2022961
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

def run_training(data_type="bottle",
                 model_dir="logs",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device="cuda",
                 workers=8,
                 size=256,
                 save_interval=1000,
                 args = None):
    weight_decay = 0.00003
    momentum = 0.9
    # augmentation:
    min_scale = 1

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(transforms.Resize((size, size)))
    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    train_data = MVTecAT(args.data_dir, data_type, transform=train_transform, size=int(size * (1 / min_scale)))
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, prefetch_factor=5)

    # create Model:
    head_layers = [512] * head_layer + [128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()
    model.train()
    loss_fn = paddle.nn.CrossEntropyLoss()
    loss_sspcab = paddle.nn.MSELoss()
    if optim_name == "sgd":
        scheduler = CosineAnnealingDecay(learning_rate=learninig_rate, T_max=65536)
        optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=momentum,
                                   weight_decay=weight_decay)
        # scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out

    dataloader_inf = get_data_inf()

    total_loss = 0
    total_reader_cost = 0
    t0 = time.time()
    max_auroc = 0
    # f_epoch_auroc = open("%s/%s/epoch")
    for epoch in range(epochs):
        if epoch == freeze_resnet:
            model.unfreeze()

        batch_embeds = []
        t1 = time.time()
        batch_idx, data = next(dataloader_inf)

        xs = [x for x in data]
        # zero the parameter gradients
        optimizer.clear_grad()
        xc = paddle.concat(xs, axis=0)
        total_reader_cost += time.time() - t1
        embeds, logits,in_sspcab,out_sspcab = model(xc)

        # calculate label
        y = paddle.arange(len(xs))  # need transform to cuda
        y = paddle.repeat_interleave(y, xs[0].shape[0], 0)
        loss = loss_fn(logits, y) + 0.1*loss_sspcab(in_sspcab,out_sspcab)
        # regulize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with paddle.no_grad():
            predicted = paddle.argmax(logits, axis=1)
            # print(logits)
            # print(predicted)
            # print(y)
            # print(predicted.shape[0])
            accuracy = paddle.divide(paddle.sum(predicted == y), paddle.to_tensor(predicted.shape[0]))

            if test_epochs>0 and epoch%test_epochs == 0:
                batch_embeds = []
                batch_embeds.append(embeds.cpu().detach())
                model.eval()
                roc_auc = eval_model("", data_type, device=device,
                                     save_plots=False,
                                     size=size,
                                     show_training_data=False,
                                     model=model)
                if roc_auc > max_auroc:
                    max_auroc = roc_auc
                    paddle.save(model.state_dict(), os.path.join(str(model_dir),data_type,
                                                             "final.pdparams" % epoch))
                # train_embed=batch_embeds)
                model.train()
            if epoch % args.log_interval == 0:
                total_bacth_cost = time.time() - t0

                print("epoch:%d/%d loss:%.4f acc:%.3f avg_reader_cost:%.3f avg_batch_cost:%.3f avg_ips:%.3f lr:%.6f"%(
                epoch+1,epochs,loss,accuracy,total_reader_cost/(epoch+1),total_bacth_cost/(epoch+1),total_bacth_cost/(epoch+1)/batch_size,optimizer.get_lr()
                ))
            # if epoch % save_interval == 0 and epoch >0:
            #     paddle.save(model.state_dict(), os.path.join(str(model_dir),data_type,
            #                                                  "%d.pdparams" % epoch))
    if test_epochs<0:
        paddle.save(model.state_dict(), os.path.join(str(model_dir),data_type,"final.pdparams"))


if __name__ == '__main__':
    # place = paddle.CUDAPlace(0)
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--epochs', default=256, type=int,help='number of epochs to train the model , (default: 256)')
    parser.add_argument('--model_dir', default="logs",help='output folder of the models , (default: models)')
    parser.add_argument('--data_dir', default="Data",help='path of data , (default: Data)')
    parser.add_argument('--no_pretrained', dest='pretrained', default=True, action='store_false',help='use pretrained values to initalize ResNet18 , (default: True)')
    parser.add_argument('--test_epochs', default=-1, type=int,help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--freeze_resnet', default=20, type=int,help='number of epochs to freeze resnet (default: 20)')
    parser.add_argument('--lr', default=0.03, type=float,help='learning rate (default: 0.03)')
    parser.add_argument('--optim', default="sgd",help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
    parser.add_argument('--batch_size', default=32, type=int,help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')
    parser.add_argument('--head_layer', default=1, type=int,help='number of layers in the projection head (default: 1)')
    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'],help='cutpaste variant to use (dafault: "3way")')
    parser.add_argument('--cuda', default=True,help='use cuda for training (default: False)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")
    parser.add_argument('--save_interval', default=1000, type=int, help="number of epochs between each model save (default:1000)")
    parser.add_argument('--log_interval', default=1, type=int, help="number of step between each log print (default:1)")
    parser.add_argument('--output', default=None, help="no sense")
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
    print(args)
    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion}
    variant = variant_map[args.variant]

    device = "cuda" if args.cuda in ["True","1","y",True] else "cpu"
    print(f"using device: {device}")

    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "a") as f:
        f.write(str(args)+"\n")

    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers,
                     save_interval=args.save_interval,
                     args=args)

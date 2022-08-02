# This code is released under the CC BY-SA 4.0 license.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F



# Squeeze and Excitation block
class SELayer(nn.Layer):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias_attr=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias_attr=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.shape

        squeeze_tensor = paddle.reshape(input_tensor,[batch_size,num_channels,H*W]).mean(axis=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.shape
        fc_out_2 = paddle.reshape(fc_out_2,[a,b,1,1])
        output_tensor = paddle.multiply(input_tensor,fc_out_2)
        return output_tensor


# SSPCAB implementation
class SSPCAB(nn.Layer):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2D(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2D(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2D(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2D(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x


class head(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(head, self).__init__()


class BatchNorm1D_new(paddle.nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)

class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(ProjectionNet, self).__init__()
        self.resnet18 = paddle.vision.models.resnet18(pretrained=pretrained)
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(BatchNorm1D_new(num_neurons,affine=True,track_running_stats=True))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons
        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes, bias_attr=True)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True



class ProjectionNet_sspcab(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(ProjectionNet_sspcab, self).__init__()
        #将resnet拆散，便于替换倒数第二层卷积层
        self.resnet18 = paddle.vision.models.resnet18(pretrained=pretrained,num_classes=0)
        # print(self.resnet18)
        self.resnet18.layer4 = nn.Identity()
        self.resnet18.avgpool = nn.Identity()
        # self.resnet18.fc = nn.Identity()
        downsample = nn.Sequential(nn.Conv2D(256, 512, kernel_size=[1, 1], stride=[2, 2], data_format="NCHW"),
                                        nn.BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05))
        self.BasicBlock0 = paddle.vision.models.resnet.BasicBlock(256,512,stride=2,downsample=downsample,
                                                                  norm_layer=nn.BatchNorm2D)
        # print("BB0:",self.BasicBlock0)
        self.BasicBlock1_conv1 = nn.Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format="NCHW")
        self.BasicBlock1_bn1 = nn.BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
        self.sspcab = SSPCAB(512)
        self.BasicBlock1_conv2 = nn.Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format="NCHW")
        self.BasicBlock1_bn2 = nn.BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(BatchNorm1D_new(num_neurons))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons
        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes, bias_attr=True)

    def forward(self, x):
        #拆散resnet
        x = self.resnet18(x)
        in_sspcab = self.BasicBlock0(x)
        identity = in_sspcab
        #将倒数第二层卷积层替换为SSPCAB
        out_sspcab = self.sspcab(in_sspcab)
        # x = self.BasicBlock1_conv1(x)
        x = self.BasicBlock1_bn1(out_sspcab)
        x = nn.functional.relu(x)
        x = self.BasicBlock1_conv2(x)
        x = self.BasicBlock1_bn2(x)
        x += identity
        x = nn.functional.relu(x)
        x = self.avgpool(x)
        embeds = paddle.flatten(x,1)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits,in_sspcab,out_sspcab

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True


# Example of how our block should be updated
# mse_loss = nn.MSELoss()
# cost_sspcab = mse_loss(input_sspcab, output_sspcab)
if __name__ == '__main__':
    pnet = ProjectionNet(False,[512,128])
    print(pnet)

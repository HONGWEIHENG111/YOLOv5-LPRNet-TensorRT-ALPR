import torch.nn as nn
import torch
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

'''
新增：原作者在代码中写入了 nn.MaxPool3d，但他传入的图像张量明明是 4 维的 [batch_size, Channels, Height, Width]。
在 PyTorch 中，如果你给 3D 池化层喂 4 维数据，PyTorch 会宽容地把它当成没有批次维度 (Unbatched) 的数据来处理，即强行把它理解为 [Channels, Depth, Height, Width]。
这样一来，原作者就极其讨巧地利用 3D 池化的深度（Depth）步长，实现了对通道数 (Channels) 的池化降维（比如通过 stride=(2, 1, 2) 把 128 通道降到了 64 通道）。

但是，TensorRT 是强类型且极度严谨的。它绝对不允许这种“指鹿为马”的维度歧义操作，它要求 3D 池化必须老老实实地输入 5 维张量 [N, C, D, H, W]，因此直接抛出了 at least 5 dimensions are required 的错误。

🛠️ 终极修改方案：封装一个 TRT 专属的 5D 降维层
我们不能简单地把它们改成 2D 池化，因为那样会丢失原作者“通道降维”的数学逻辑，导致网络前向传播报错。

最完美的解法是：写一个小包装类，在进入池化前把 4D 升维成 5D 满足 TensorRT 的强迫症，池化完后再降回 4D。 这个操作对 ONNX 和 TensorRT 是完全透明且高度友好的。
'''


class TRT_MaxPool3d(nn.Module):
    """
    彻底解决 TensorRT 维度报错的最佳方案！
    利用数学等效，将非标准的 3D 池化拆解为 '通道切片(Slice) + 2D池化(MaxPool2d)'。
    """

    def __init__(self, kernel_size, stride):
        super(TRT_MaxPool3d, self).__init__()
        # 提取通道维度的步长 (原代码中分别是 1, 2, 4)
        self.channel_stride = stride[0]

        # 将空间维度的参数传给标准的 2D 池化
        self.pool = nn.MaxPool2d(kernel_size=kernel_size[1:], stride=stride[1:])

    def forward(self, x):
        # 1. 模拟 3D 池化中的通道步长降维 (跨通道切片)
        if self.channel_stride > 1:
            # 格式: [batch, channels, height, width]，按 channel_stride 跳跃采样通道
            x = x[:, ::self.channel_stride, :, :]

        # 2. 进行完全合规的标准的 2D 空间池化
        return self.pool(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),    # 0  [bs,3,24,94] -> [bs,64,22,92]
            nn.BatchNorm2d(num_features=64),                                       # 1  -> [bs,64,22,92]
            nn.ReLU(),                                                             # 2  -> [bs,64,22,92]
            TRT_MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),                 # 3  -> [bs,64,20,90]
            small_basic_block(ch_in=64, ch_out=128),                               # 4  -> [bs,128,20,90]
            nn.BatchNorm2d(num_features=128),                                      # 5  -> [bs,128,20,90]
            nn.ReLU(),                                                             # 6  -> [bs,128,20,90]
            TRT_MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),                #替换下方一行内容
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),                 # 7  -> [bs,64,18,44]
            small_basic_block(ch_in=64, ch_out=256),                               # 8  -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),                                      # 9  -> [bs,256,18,44]
            nn.ReLU(),                                                             # 10 -> [bs,256,18,44]
            small_basic_block(ch_in=256, ch_out=256),                              # 11 -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),                                      # 12 -> [bs,256,18,44]
            nn.ReLU(),                                                             # 13 -> [bs,256,18,44]
            TRT_MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),                #替换下方一行内容
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),                 # 14 -> [bs,64,16,21]
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                          # 15 -> [bs,64,16,21]
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),   # 16 -> [bs,256,16,18]
            nn.BatchNorm2d(num_features=256),                                            # 17 -> [bs,256,16,18]
            nn.ReLU(),                                                                   # 18 -> [bs,256,16,18]
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                                  19 -> [bs,256,16,18]
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # class_num=68  20  -> [bs,68,4,18]
            nn.BatchNorm2d(num_features=class_num),                                             # 21 -> [bs,68,4,18]
            nn.ReLU(),                                                                          # 22 -> [bs,68,4,18]
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )
        # self.connected = nn.Sequential(
        #     nn.Linear(class_num * 88, 128),
        #     nn.ReLU(),
        # )
        #

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        # keep_features: [bs,64,22,92]  [bs,128,20,90] [bs,256,18,44] [bs,68,4,18]
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                # [bs,64,22,92] -> [bs,64,4,18]
                # [bs,128,20,90] -> [bs,128,4,18]
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                # [bs,256,18,44] -> [bs,256,4,18]
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)

            # 没看懂这是在干嘛？有上面的avg提取上下文信息不久可以了？
            f_pow = torch.pow(f, 2)     # [bs,64,4,18]  所有元素求平方
            f_mean = torch.mean(f_pow)  # 1 所有元素求平均
            f = torch.div(f, f_mean)    # [bs,64,4,18]  所有元素除以这个均值
            global_context.append(f)

        x = torch.cat(global_context, 1)  # [bs,516,4,18]
        x = self.container(x)  # -> [bs, 68, 4, 18]   head头
        logits = torch.mean(x, dim=2)  # -> [bs, 68, 18]  # 68 字符类别数   18字符序列长度

        return logits




    # https://blog.csdn.net/weixin_39027619/article/details/106143755
    # def forward(self, x):
    #     x = self.backbone(x)
    #     pattern = x.flatten(1, -1)
    #     pattern = self.connected(pattern)
    #     width = x.size()[-1]
    #     pattern = torch.reshape(pattern, [-1, 128, 1, 1])
    #     pattern = pattern.repeat(1, 1, 1, width)
    #     x = torch.cat([x, pattern], dim=1)
    #     x = self.container(x)
    #     logits = x.squeeze(2)
    #     return logits


# def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
#
#     Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
#
#     if phase == "train":
#         return Net.train()
#     else:
#         return Net.eval()

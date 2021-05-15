<<<<<<< HEAD
from torch import nn

# 残差块
class resblock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(resblock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.res = nn.Sequential(

            nn.Conv3d(self.in_channel, self.out_channel, (2, 3, 3), stride=(1, self.stride, self.stride), padding=1),
            nn.Conv3d(self.out_channel, self.out_channel, (2, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(True)
        )
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, self.out_channel, (1, 1, 1), stride=(1, self.stride, self.stride))
            )
        else:
            self.downsample = lambda x: x


    def forward(self, input):

        output = self.res(input)
        sample = self.downsample(input)
        output = output + sample
        return output

# 动作的模型
class actNet(nn.Module):

    def __init__(self, opt):
        super(actNet, self).__init__()
        self.actdim = opt.actdim
        self.net = nn.Sequential(
            # 卷积
            nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2), stride=1),
        )
        self.build_resblock(64, 64, 2)
        self.build_resblock(64, 80, 2, 2)
        self.build_resblock(80, 128, 2, 2)
        self.build_resblock(128, 200, 2, 2)
        self.net = nn.Sequential(
            self.net,
            nn.AdaptiveAvgPool3d(1)
            # nn.Linear(200, 1)
        )
        self.line = nn.Linear(200, self.actdim)
        #self.resBlock_1 = resblock(64, 64)
        #self.resBlock_2 = resblock(64, 80, 2)
        #self.resBlock_3 = resblock(80, 128, 2)

    def forward(self, input):

        input = input.unsqueeze(0)
        output = self.net(input)
        output = output.view(1, 200)
        output = self.line(output)
        return output

    def build_resblock(self, in_channel, out_channel, block, stride=1):
        self.net = nn.Sequential(
            self.net,
            resblock(in_channel, out_channel, stride),
            nn.ReLU(True)
        )
        for i in range(1, block):
            self.net = nn.Sequential(
                self.net,
                resblock(out_channel, out_channel, stride),
                nn.ReLU(True)
            )




# 动作的模型
class moveNet(nn.Module):

    def __init__(self, opt):
        super(moveNet, self).__init__()
        self.movedim = opt.movedim
        self.net = nn.Sequential(
            # 卷积
            nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2), stride=1),
        )
        self.build_resblock(64, 64, 2)
        self.build_resblock(64, 80, 2, 2)
        self.build_resblock(80, 128, 2, 2)
        self.build_resblock(128, 200, 2, 2)
        self.net = nn.Sequential(
            self.net,
            nn.AdaptiveAvgPool3d(1)
            # nn.Linear(200, 1)
        )
        self.line = nn.Linear(200, self.movedim)
        # self.resBlock_1 = resblock(64, 64)
        # self.resBlock_2 = resblock(64, 80, 2)
        # self.resBlock_3 = resblock(80, 128, 2)

    def forward(self, input):
        input = input.unsqueeze(0)
        output = self.net(input)
        output = output.view(1, 200)
        output = self.line(output)
        return output

    def build_resblock(self, in_channel, out_channel, block, stride=1):
        self.net = nn.Sequential(
            self.net,
            resblock(in_channel, out_channel, stride),
            nn.ReLU(True)
        )
        for i in range(1, block):
            self.net = nn.Sequential(
                self.net,
                resblock(out_channel, out_channel, stride),
                nn.ReLU(True)
            )





=======
from torch import nn

# 残差块
class resblock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(resblock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.res = nn.Sequential(

            nn.Conv3d(self.in_channel, self.out_channel, (2, 3, 3), stride=(1, self.stride, self.stride), padding=1),
            nn.Conv3d(self.out_channel, self.out_channel, (2, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(True)
        )
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, self.out_channel, (1, 1, 1), stride=(1, self.stride, self.stride))
            )
        else:
            self.downsample = lambda x: x


    def forward(self, input):

        output = self.res(input)
        sample = self.downsample(input)
        output = output + sample
        return output

# 动作的模型
class actNet(nn.Module):

    def __init__(self, opt):
        super(actNet, self).__init__()
        self.actdim = opt.actdim
        self.net = nn.Sequential(
            # 卷积
            nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2), stride=1),
        )
        self.build_resblock(64, 64, 2)
        self.build_resblock(64, 80, 2, 2)
        self.build_resblock(80, 128, 2, 2)
        self.build_resblock(128, 200, 2, 2)
        self.net = nn.Sequential(
            self.net,
            nn.AdaptiveAvgPool3d(1)
            # nn.Linear(200, 1)
        )
        self.line = nn.Linear(200, self.actdim)
        #self.resBlock_1 = resblock(64, 64)
        #self.resBlock_2 = resblock(64, 80, 2)
        #self.resBlock_3 = resblock(80, 128, 2)

    def forward(self, input):

        input = input.unsqueeze(0)
        output = self.net(input)
        output = output.view(1, 200)
        output = self.line(output)
        return output

    def build_resblock(self, in_channel, out_channel, block, stride=1):
        self.net = nn.Sequential(
            self.net,
            resblock(in_channel, out_channel, stride),
            nn.ReLU(True)
        )
        for i in range(1, block):
            self.net = nn.Sequential(
                self.net,
                resblock(out_channel, out_channel, stride),
                nn.ReLU(True)
            )




# 动作的模型
class moveNet(nn.Module):

    def __init__(self, opt):
        super(moveNet, self).__init__()
        self.movedim = opt.movedim
        self.net = nn.Sequential(
            # 卷积
            nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2), stride=1),
        )
        self.build_resblock(64, 64, 2)
        self.build_resblock(64, 80, 2, 2)
        self.build_resblock(80, 128, 2, 2)
        self.build_resblock(128, 200, 2, 2)
        self.net = nn.Sequential(
            self.net,
            nn.AdaptiveAvgPool3d(1)
            # nn.Linear(200, 1)
        )
        self.line = nn.Linear(200, self.movedim)
        # self.resBlock_1 = resblock(64, 64)
        # self.resBlock_2 = resblock(64, 80, 2)
        # self.resBlock_3 = resblock(80, 128, 2)

    def forward(self, input):
        input = input.unsqueeze(0)
        output = self.net(input)
        output = output.view(1, 200)
        output = self.line(output)
        return output

    def build_resblock(self, in_channel, out_channel, block, stride=1):
        self.net = nn.Sequential(
            self.net,
            resblock(in_channel, out_channel, stride),
            nn.ReLU(True)
        )
        for i in range(1, block):
            self.net = nn.Sequential(
                self.net,
                resblock(out_channel, out_channel, stride),
                nn.ReLU(True)
            )





>>>>>>> d00f7fee553731b8aa5db54bf247e6437c5d6052

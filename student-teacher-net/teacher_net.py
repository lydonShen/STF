import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from tqdm import tqdm
import math
from PIL import Image
from torchvision import transforms

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, if_include_top=False):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(p=0.5)

        if if_include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.if_include_top = if_include_top

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x =self.dropout(x)
        out3 = self.layer2(x)
        out3 = self.dropout(out3)
        return out3

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # print("conv2dbn")
        # print(x.shape)
        return nn.functional.relu(x)

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, with_conv_shortcut=True, dropout_prob=0.3):
        super(Conv_Block, self).__init__()
        self.conv1 = Conv2d_BN(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.conv2 = Conv2d_BN(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=stride) if with_conv_shortcut else None
        self.bn_shortcut = nn.BatchNorm2d(out_channels) if with_conv_shortcut else None
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        shortcut = x
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = x + shortcut
        # print("conv")
        # print(x.shape)
        x = self.dropout(x)
        return nn.functional.relu(x)

class SiameseTeacherModel(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(SiameseTeacherModel, self).__init__()
        self.teacher = ResNet(BasicBlock, [2, 2, 2, 2], if_include_top=False)
        self.combined = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv_Block(32, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2, 1)
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.teacher.load_state_dict(torch.load('./backbone.pth'), strict=False)


    def forward(self, input1, input2):
        output1 = self.teacher.forward(input1)
        output2 = self.teacher.forward(input2)
        combined = torch.cat([output1, output2], dim=1)
        output = self.combined(combined)
        output = output.squeeze()
        return output

def load_data(data_dir, loss_dir):
    x = []
    y = []
    dataset = []
    with open(loss_dir) as file:
        lines = file.readlines()
    benefitdict = {}
    for i in lines:
        replace = i.split(",")
        benefitdict[replace[0]] = float(replace[1])

    for statename, label in benefitdict.items():
        img_path = data_dir + "//" + str(statename) + ".jpg" 
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor
        dataset.append((img_tensor, label))
    return dataset

class TeacherDataset(Dataset):
    def __init__(self, data_dir, loss_dir, is_train):
        self.dataset = load_data(data_dir, loss_dir)
        self.num_classes = 12 if is_train else 3
        self.num_samples_per_class = 250
        self.combinations = self.generate_combinations()
        self.is_train=is_train
        self.count_0 = 0
        self.count_1 = 0

    def generate_combinations(self):
        combinations_per_class = []
        for class_idx in range(self.num_classes):
            class_start_idx = class_idx * self.num_samples_per_class
            class_end_idx = (class_idx + 1) * self.num_samples_per_class
            class_indices = list(range(class_start_idx, class_end_idx))
            class_combinations = list(combinations(class_indices, 2))
            combinations_per_class.extend(class_combinations)
        return combinations_per_class

    def get_count(self):
        print("count0:", self.count_0)
        print("count1:", self.count_1)

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]
        x1,y1 = self.dataset[idx1]
        x2,y2 = self.dataset[idx2]
        y = 0 if y1 >= y2 else 1
        y = torch.tensor([y], dtype=torch.float32).squeeze()
        return x1, x2, y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainer(tarindataloader, testdataloader,epochs=10, learning_rate=0.001, weight_decay=None):
    device_ids = [0,1]
    torch.cuda.set_device(device_ids[0])

    model = SiameseTeacherModel().to(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=0, last_epoch=-1)

    for epoch in range(epochs):
        tarinDataloader = DataLoader(tarindataloader, batch_size=512, shuffle=True)
        model.train()
        total_loss = 0.0
        total_correct_train = 0
        total_samples_train = 0
        pbar = tqdm(enumerate(tarinDataloader), total=len(tarinDataloader))
        for it, batch in pbar:
            inputs1, inputs2, labels = batch
            inputs1 = inputs1.to(device_ids[0])
            inputs2 = inputs2.to(device_ids[0])
            labels = labels.to(device_ids[0])

            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predictions = torch.sigmoid(outputs) >= 0.5
            total_correct_train += (predictions == labels).sum().item()
            total_samples_train += len(labels)

            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {learning_rate:e}")
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join("saved_models", f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    final_checkpoint_path = os.path.join("saved_models", "final_model.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved at {final_checkpoint_path}")
    return model

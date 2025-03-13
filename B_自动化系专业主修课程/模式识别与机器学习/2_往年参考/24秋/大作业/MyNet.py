'''
本文件定义了ResNet网络结构，包括BasicBlock和ResNet两个类，以及ImageClassifier类，用于训练和验证模型。
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import MyDataLoader, MyImageDataset

'''
BasicBlock类定义了ResNet的基本残差块，包括3个卷积层和1个残差连接。
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 卷积层 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 卷积层 3
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # 残差连接的下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接
        out = self.relu(out)
        return out

'''
ResNet类定义了ResNet网络结构，包括初始卷积层、残差层、全连接层等
''' 
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # 权重初始化
        self.apply(self._init_weights)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
'''
ImageClassifier类用于训练和验证模型
'''
class ImageClassifier:
    def __init__(self, train_folder, val_folder, batch_size=64, learning_rate=0.1, num_epochs=100):
        self.device = torch.device("cuda")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_folder = train_folder
        self.val_folder = val_folder

        # 数据增强和预处理
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 建立统一的类别索引映射
        self.class_to_idx = self.get_class_to_idx([train_folder, val_folder])

        # 加载数据集并获取类别数量
        self.train_dataset = MyImageDataset(train_folder, self.train_transforms, self.class_to_idx)
        self.val_dataset = MyImageDataset(val_folder, self.val_transforms, self.class_to_idx)
        self.num_classes = len(self.class_to_idx)

        self.train_loader = MyDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = MyDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

    def get_class_to_idx(self, folder_paths):
        classes = set()
        for folder_path in folder_paths:
            class_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            classes.update(class_dirs)
        classes = sorted(list(classes))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return class_to_idx

    def build_model(self):
        return ResNet(BasicBlock, [3, 3, 3], self.num_classes)

    def train(self):
        ac = []
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                acc_test , loss_test= self.validate()
                ac.append([epoch,
                           epoch_loss,
                            accuracy,
                            loss_test,
                            acc_test])
        with open('training_log_3layer.txt', 'w') as f:
            for record in ac:
                f.write(f'Epoch: {record[0]},loss_train:{record[1]:.4f} ,acc_train: {record[2]:.4f}%,loss_test:{record[3]:.4f} ,acc_test: {record[4]:.4f}\n')
            # 保存模型参数
        save_model(self.model, 'resnet_model.pth')

    def validate(self):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = val_loss / len(self.val_loader.dataset)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        return accuracy,epoch_loss
    
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

if __name__ == "__main__":

    train_folder = 'imagenet_mini/train'
    val_folder = 'imagenet_mini/val'
    classifier = ImageClassifier(train_folder, val_folder, batch_size=64, learning_rate=0.1, num_epochs=3000)
    classifier.train()

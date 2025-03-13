'''
本文件主要包含了任务1的数据加载器和混淆矩阵绘制函数，以及任务2的数据加载器、数据集类。
'''
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from torch.utils.data import Dataset
import math
from torch.optim import Optimizer


'''
任务1的数据加载器
'''
def load_features(folder_path, selected_classes, device='cpu'):
    features, labels = [], []
    
    for label_idx, class_dir in enumerate(selected_classes):
        class_path = os.path.join(folder_path, class_dir)
        
        for file in os.listdir(class_path):
            if file.endswith('.pt'):
                feature_path = os.path.join(class_path, file)
                feature = torch.load(feature_path, map_location=device)
                features.append(feature.cpu().numpy().flatten())  
                labels.append(label_idx)  
                
    return features, labels

'''
任务1的混淆矩阵
'''
def metrics(y_p, y_v):
    # 计算准确率
    acc = np.mean(y_p == y_v)
    print(f"准确率为 {acc:.2f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_v, y_p)
    labels = set(np.concatenate((y_v, y_p)))  # 提取所有唯一类别标签
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

'''
任务一的可视化函数
'''
def pltlossandacc(epochs, loss, acc):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Loss", linestyle='-', color='blue')
    plt.plot(epochs, acc, label="Validation Accuracy", linestyle='--', color='green')
    plt.title("Loss and Validation Accuracy over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
任务2的数据集类
'''
class MyImageDataset(Dataset):
    def __init__(self, folder_path, transform, class_to_idx):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = class_to_idx

        for class_name, label_idx in self.class_to_idx.items():
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                if file.endswith(('.JPEG')):
                    image_path = os.path.join(class_path, file)
                    self.images.append(image_path)
                    self.labels.append(label_idx)


    def __len__(self):
        return len(self.images)  # 返回数据集中样本的数量s
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img, label
    


'''
任务2的数据集加载器
'''
class MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        start_index = self.current_index
        end_index = min(self.current_index + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_index:end_index]
        self.current_index = end_index

        batch_data = [self.dataset[i] for i in batch_indices]
        inputs, labels = zip(*batch_data)
        return torch.stack(inputs), torch.tensor(labels)
    
'''
自定义SGD优化器
'''

class MySGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [torch.zeros_like(p) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data + self.weight_decay * p.data
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            p.data += self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data.zero_()
# class MySGD(Optimizer):
#     def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
#         super(MySGD, self).__init__(params, defaults)

#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if group['weight_decay'] != 0:
#                     grad = grad + group['weight_decay'] * p.data
#                 if group['momentum'] != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(group['momentum']).add_(grad)
#                     grad = buf
#                 p.data.add_(-group['lr'], grad)

        # return loss
        

'''
自定义Adam优化器
'''
class MyAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentums = [torch.zeros_like(p, device=p.device) for p in params]
        self.velocities = [torch.zeros_like(p, device=p.device) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * p.data
                
                self.momentums[i] = self.beta1 * self.momentums[i] + (1 - self.beta1) * grad
                self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.momentums[i] / (1 - self.beta1 ** self.t)
                v_hat = self.velocities[i] / (1 - self.beta2 ** self.t)

                p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

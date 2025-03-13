'''
本文件实现任务1传统方法
'''
import os
import random
import argparse
import numpy as np

from utils import load_features
from utils import metrics
from utils import pltlossandacc

from cvxopt import matrix, solvers

'''
实现了基于逻辑回归的多分类
'''
class MyLogistic():
    def __init__(self, x_train, y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.k = y_train.shape[1]
        self.n = x_train.shape[1]
        self.wT = np.random.rand(x_train.shape[1],y_train.shape[1])*0.1
        self.b = np.zeros((1, self.k))
        self.ac = []
        self.loss = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z,axis=1,keepdims=True))
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    def cross_loss(self,W,x,Y,b):
        z = np.dot(x,W) + b
        eps = 1e-12
        soft_z = self.softmax(z) + eps
        loss = - np.sum(Y * np.log(soft_z)) / x.shape[0]
        return loss

    def fit(self,iters=1000, lr = 1e-1):
        self.ac = []
        self.loss = []
        for i in range(iters):
            z = np.dot(self.x_train,self.wT)+self.b
            p = self.softmax(z)
            g_wT = np.dot(self.x_train.T, (p-self.y_train))/self.x_train.shape[0]
            g_b = np.mean(p-self.y_train,axis=0,keepdims=True)

            self.wT -= lr * g_wT
            self.b -= lr * g_b
            loss = self.cross_loss(self.wT,self.x_train,self.y_train,self.b)
            
            if i % 10 == 0 :
                print(f"loss{loss:.4f},iters{i}")
                y_pred = self.predict(self.x_test)
                if self.y_test.ndim == 1:
                    acc = np.mean(y_pred == self.y_test)
                else:
                    acc = np.mean(y_pred == np.argmax(self.y_test, axis=1))

                self.ac.append(acc)
                self.loss.append(loss)
            
                
                

    def predict(self,x_test):
        z = np.dot(x_test,self.wT)+self.b
        p = self.softmax(z)
        return np.argmax(p,axis=1)
    
'''
实现了基于svm的二分类
'''
class mysvm():
    def __init__(self,  C=1.0):
        self.w = None
        self.b = 0
        self.C = C

    def fit(self, X, y):
        n_s = X.shape[0]
        self.X = X
        self.y = y

        K = np.zeros((n_s, n_s))
        for i in range(n_s):
            for j in range(n_s):
                K[i, j] = self.kernelfunction(X[i], X[j])
        # 使用cvxopt求解二次规划问题
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_s))
        G = matrix(np.vstack((-np.eye(n_s), np.eye(n_s))))
        h = matrix(np.hstack((np.zeros(n_s), np.ones(n_s) * self.C)))
        A = matrix(y.astype(float), (1, n_s))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)

        self.alpha = np.ravel(solution['x'])

        s_v = self.alpha > 1e-5
        self.s_vs = X[s_v]
        self.s_v_ls = y[s_v]
        self.alpha = self.alpha[s_v]

        self.b = 0
        for i in range(len(self.alpha)):
            self.b += self.s_v_ls[i]
            self.b -= np.sum(
                self.alpha * self.s_v_ls *
                K[s_v][i][s_v]
            )
        self.b /= len(self.alpha)

        self.w = np.sum((self.alpha * self.s_v_ls)[:, None] * self.s_vs, axis=0)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def decision(self, X):
        decision = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(len(self.alpha)):
                decision[i] += self.alpha[j] * self.s_v_ls[j] * self.kernelfunction(self.s_vs[j], X[i])
        return decision + self.b

    def kernelfunction(self, x1, x2):
        return np.dot(x1, x2)


'''
另外实现了基于svm的多分类
'''
class MySVM:
    def __init__(self, C=1.0):
        self.C = C
        self.models = {}  # 用于存储每个类别的二分类器

    def fit(self, X, y):
        """
        训练 One-vs-All 多分类 SVM
        """
        self.classes = np.unique(y)  # 获取所有类别
        for c in self.classes:
            # 创建新的二分类标签：当前类别为 +1，其他类别为 -1
            binary_labels = np.where(y == c, 1, -1)
            # 初始化二分类器
            model = mysvm( C=self.C)
            # 训练当前类别的二分类器
            model.fit(X, binary_labels)
            self.models[c] = model  # 保存二分类器

    def predict(self, X):
        # 收集每个类别的决策函数值
        decision_values = {}
        for c, model in self.models.items():
            decision_values[c] = model.decision(X)

        # 对每个样本，选择决策函数值最大的类别
        decisions = np.vstack([decision_values[c] for c in self.classes]).T
        predictions = self.classes[np.argmax(decisions, axis=1)]
        return predictions

def MyBayes(X_train, y_train, X_test):
    classes = np.unique(y_train)
    means = np.array([np.mean(X_train[y_train == c], axis=0) for c in classes])
    dists = []
    for mean in means:
        diff = X_test - mean
        dis = np.sum(diff ** 2, axis=1)
        dists.append(dis)
    dists = np.array(dists)
    y_pred = classes[np.argmin(dists, axis=0)]
    return y_pred

'''
训练和预测
'''
def train_pred(classifier_opt="svm"):
    t_path = 'imagenet_mini/train'
    v_path = 'imagenet_mini/val'
    # 从训练集中选择10个类别
    a_cls = [d for d in os.listdir(t_path) if os.path.isdir(os.path.join(t_path, d))]
    s_cls = random.sample(a_cls, 10)

    # 加载训练和验证数据
    t_f, t_ls = load_features(t_path, s_cls)
    v_f, v_ls = load_features(v_path, s_cls)
    t_f = np.array(t_f)
    v_f = np.array(v_f)
    t_ls = np.array(t_ls)
    v_ls = np.array(v_ls)

    # 根据选择的分类器类型初始化分类器
    if classifier_opt == "bayes":
        clf = MyBayes(t_f, t_ls, v_f)
        y_pred = clf
    elif classifier_opt == "logistic":
        #先转化未独热编码
        y_one = np.zeros((t_ls.shape[0], len(np.unique(t_ls))))
        for i in range(len(t_ls)):
            y_one[i, t_ls[i]] = 1
        clf = MyLogistic(t_f, y_one, v_f, v_ls)
        clf.fit()
        y_pred = clf.predict(v_f)
        loss_epochs = range(0, 1000, 10)
        pltlossandacc(loss_epochs, clf.loss, clf.ac)

    elif classifier_opt == "svm":
        clf = MySVM(C=1.0)
        # 训练分类器
        clf.fit(t_f, t_ls)
        y_pred = clf.predict(v_f)

    metrics(y_pred, v_ls)
    # 随机采样的类别
    print(f"Classes selected: {s_cls}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="logistic", choices=[ "svm", "logistic","bayes"])
    args = parser.parse_args()
    train_pred(classifier_opt=args.opt)
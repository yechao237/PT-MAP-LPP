import time
import math
import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch

from numpy.linalg import norm
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

use_gpu = torch.cuda.is_available()


def My_sinkhornKnopp(A, maxiter, tol):
    A = np.asarray(A)
    iter = 1
    c = 1. / A.sum(axis=0)
    r = 1. / (A @ c)
    while iter < maxiter:
        iter = iter + 1
        cinv = r.dot(A)
        if np.max(np.abs(cinv * c - 1)) <= tol:
            break
        c = 1. / cinv
        r = 1. / (A @ c)
    A = A * (r[:, np.newaxis] * c)
    return A


def Unsupervised_k_means(data, n_lsamples):
    # 使用k-means将数据分成5组（即5个类），将5个有标签的数据作为初始类心
    if n_lsamples == 25:
        initial_centroids = np.array([data[:25, :][i::5].mean(axis=0) for i in range(5)])
    else:
        initial_centroids = data[:n_lsamples, :]
    kmeans = KMeans(n_clusters=5, init=initial_centroids, n_init=1)

    kmeans.fit(data)

    # 使用predict方法预测每个数据点属于哪个类
    predictions = kmeans.predict(data)
    # print(predictions)

    # 将伪标签转换为one-hot编码
    one_hot_labels = np.eye(5)[predictions[n_lsamples:]]

    # 使用softmax函数计算每个类的概率
    probabilities = np.exp(one_hot_labels) / np.sum(np.exp(one_hot_labels), axis=1, keepdims=True)

    # 计算伪标签准确率
    # true_labels = np.tile([0, 1, 2, 3, 4], 15)
    # accuracy = np.mean(predictions[n_lsamples:] == true_labels)
    # print("伪标签准确率：", accuracy)

    return predictions[n_lsamples:], probabilities


def get_scores_knn(domainS_proj, domainT_proj, domainS_labels):
    from sklearn.neighbors import KNeighborsClassifier
    # 构造 KNeighborsClassifier 对象
    clf = KNeighborsClassifier(n_neighbors=1)
    # 在支持集上训练分类器
    clf.fit(domainS_proj, domainS_labels)
    # 在查询集上进行预测
    scores = clf.predict_proba(domainT_proj)
    return scores


def get_scores_SVM(domainS_proj, domainT_proj, domainS_labels):
    # SVM(python中为SVC) 8808%
    from sklearn.svm import SVC
    # C 为SVM算法的正则化参数
    # 构造 SVC 对象
    clf = SVC(C=1.1, kernel='rbf')
    # 在支持集上训练分类器
    clf.fit(domainS_proj, domainS_labels)
    # 在查询集上进行预测
    scores = clf.decision_function(domainT_proj)
    return scores


def get_mean_by_label(domainS_proj, domainS_labels):
    unique_labels = np.unique(domainS_labels)
    mean_projs = []

    for label in unique_labels:
        indices = np.where(domainS_labels == label)[0]
        mean_proj = np.mean(domainS_proj[indices], axis=0)
        mean_projs.append(mean_proj)

    mean_projs = np.array(mean_projs)
    return mean_projs


def get_scores_meta_LSTM(domainS_proj, domainT_proj, domainS_labels, n_shot):  # 0.8648
    from tensorflow.keras.layers import Input, LSTM, Dense, Flatten
    from tensorflow.keras.models import Model
    # 假设 domainS_proj 和 domainT_proj 分别为支持集和查询集的投影矩阵，
    # domainS_labels 为支持集数据的标签，n_way 和 n_shot 分别为分类任务中的类别数和样本数
    if n_shot == 5:
        # 对domainS_proj按照标签求均值
        domainS_proj = get_mean_by_label(domainS_proj, domainS_labels)
        # 更新domainS_labels为(0,1,2,3,4)
        domainS_labels = np.array([0, 1, 2, 3, 4])
    # 构造Meta-Learner LSTM模型
    n_way = 5
    input = Input(shape=(1, domainS_proj.shape[1]))
    x = LSTM(64)(input)
    x = Dense(n_way, activation='softmax')(x)
    model = Model(input, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练Meta-Learner LSTM模型
    X = domainS_proj.reshape(n_way, 1, -1)
    y = np.eye(n_way)[domainS_labels]
    model.fit(X, y, epochs=50, batch_size=32)
    # 在查询集上进行预测
    X = domainT_proj.reshape(-1, 1, domainT_proj.shape[1])
    scores = model.predict(X)
    return scores


def get_scores_Matching_Networks(domainS_proj, domainT_proj, domainS_labels):
    """
    使用Matching Networks获取查询集的得分。

    Args:
        domainS_proj: (5, 5) 的支持集矩阵，表示5个样本，每个样本5个特征
        domainT_proj: (75, 5) 的查询集矩阵，表示共75个样本，每个样本5个特征
        domainS_labels: (5,) 的支持集标签

    Returns:
        scores: (75, 5) 的得分矩阵，分别为查询集的75个样本对应5类的得分
    """
    # 计算查询集中每个样本与支持集中每个样本的欧几里得距离
    # (75, 5) 的矩阵，其中第 i 行第 j 列表示查询集中第 i 个样本与支持集中第 j 个样本的距离
    distances = euclidean_distances(domainT_proj, domainS_proj)
    # 将距离转换为相似度，使用高斯核函数，带宽为 1
    similarities = np.exp(-distances ** 2)
    # 将支持集标签转换为独热编码
    num_classes = len(np.unique(domainS_labels))
    one_hot_labels = np.eye(num_classes)[domainS_labels]
    # 计算查询集每个样本在每个支持集类别上的得分
    # (75, 5) 的矩阵，其中第 i 行第 j 列表示查询集中第 i 个样本在支持集第 j 个类别上的得分
    scores = np.dot(similarities, one_hot_labels)
    return scores


from torch import nn
from torch.nn import functional as F


class ProtoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProtoNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, support, query):
        """
        support: support set, shape=(num_classes, num_support, input_dim)
        query: query set, shape=(num_classes * num_queries, input_dim)
        """
        num_classes, num_support, input_dim = support.shape
        num_queries = query.shape[0] // num_classes

        # Calculate prototypes
        prototypes = support.mean(dim=1)  # shape=(num_classes, input_dim)

        # Apply nonlinear transformation to input
        support = F.relu(self.hidden(support))
        query = F.relu(self.hidden(query))

        # Calculate distances to prototypes
        prototypes = prototypes.unsqueeze(0).repeat(num_classes * num_queries, 1, 1)
        query = query.unsqueeze(1).repeat(1, num_classes, 1)
        distances = -((prototypes - query) ** 2).sum(dim=-1)

        # Calculate probabilities
        distances = distances.view(num_classes * num_queries, -1)
        probs = F.softmax(distances, dim=-1)

        # Calculate scores
        scores = self.output(support.view(num_classes * num_support, -1))
        scores = scores.view(num_classes, num_support, -1)
        scores = (probs.unsqueeze(-1) * scores).sum(dim=1)
        scores = scores.view(num_classes * num_queries, -1)

        return scores


def get_scores_ProtoNets(domainS_proj, domainT_proj, domainS_labels):
    # 计算支持集中每个类别的原型向量
    prototypes = []
    for label in np.unique(domainS_labels):
        class_samples = domainS_proj[domainS_labels == label]
        prototype = np.mean(class_samples, axis=0)
        prototypes.append(prototype)
    prototypes = np.array(prototypes)

    # 计算查询集与原型向量之间的欧几里得距离
    distances = np.sqrt(np.sum((domainT_proj[:, np.newaxis] - prototypes) ** 2, axis=2))

    # 将距离转换为相似度得分
    scores = -distances
    scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    scores = scores / np.sum(scores, axis=1, keepdims=True)

    return scores


import torch
import torch.nn as nn


# 定义ProtoNets模型
class ProtoNets(nn.Module):
    def __init__(self):
        super(ProtoNets, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_nfeat, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

    def mean_vectors(self, x, y):
        """
        计算每个类别的类均值向量
        """
        class_labels = torch.unique(y)
        num_classes = len(class_labels)
        mean_vectors = torch.zeros(num_classes, x.size(1))
        for i, c in enumerate(class_labels):
            mean_vectors[i] = torch.mean(x[y == c], dim=0)
        return mean_vectors

    def euclidean_distance(self, x, y):
        """
        计算欧氏距离
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def score(self, support, query, support_labels):
        """
        计算得分
        """
        support_encoded = self.encoder(support)
        query_encoded = self.encoder(query)
        support_mean = self.mean_vectors(support_encoded, support_labels)
        distances = self.euclidean_distance(query_encoded, support_mean)
        scores = -distances
        return scores


def get_scores_Deep_ProtoNets(domainS_proj, domainT_proj, domainS_labels, n_nfeat):

    # 转换为PyTorch的tensor
    support = torch.tensor(domainS_proj, dtype=torch.float)
    query = torch.tensor(domainT_proj, dtype=torch.float)
    support_labels = torch.tensor(domainS_labels, dtype=torch.long)

    # 实例化ProtoNets模型
    model = ProtoNets()

    # 计算得分
    scores = model.score(support, query, support_labels)

    # 转换为numpy数组并返回
    scores = scores.detach().numpy()
    return scores


def maml(model, dataloader, lr_inner=0.4, num_epochs_inner=1, num_updates_outer=1):
    """
    MAML算法实现函数，对于给定的模型和数据集，返回在数据集上的训练结果
    """
    # 在函数开始处定义device变量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_inner)
    for i in range(num_updates_outer):
        for data in dataloader:
            x, y = data
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            with torch.set_grad_enabled(True):
                for j in range(num_epochs_inner):
                    y_pred = model(x)
                    loss = F.cross_entropy(y_pred, y)
                    loss.backward()
                    optimizer.step()
    return model


def get_scores_maml(domainS_proj, domainT_proj, domainS_labels):
    """
    对于给定的支持集、查询集和标签，使用MAML算法获取查询集的得分并返回
    """
    # 在函数开始处定义device变量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在代码中使用device变量 将数据转换为PyTorch张量
    domainS_proj = torch.from_numpy(domainS_proj).float().to(device)
    domainT_proj = torch.from_numpy(domainT_proj).float().to(device)
    domainS_labels = torch.from_numpy(domainS_labels).long().to(device)

    # 定义模型和训练超参数
    num_classes = len(np.unique(domainS_labels.cpu().detach().numpy()))
    num_features = domainS_proj.shape[1]
    hidden_size = 64
    model = torch.nn.Sequential(
        torch.nn.Linear(num_features, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, num_classes)
    )
    lr_inner = 0.4
    num_epochs_inner = 1
    num_updates_outer = 1
    batch_size = 5

    # 将支持集数据转换为数据集对象
    support_set = TensorDataset(domainS_proj, domainS_labels)
    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=True)

    # 使用MAML算法对模型进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maml(model.to(device), support_loader, lr_inner, num_epochs_inner, num_updates_outer)

    # 将查询集数据转换为数据集对象
    query_set = TensorDataset(domainT_proj)
    query_loader = DataLoader(query_set, batch_size=batch_size)

    # 对查询集进行推理并返回得分
    model.eval()
    scores = []
    with torch.set_grad_enabled(False):
        for x in query_loader:
            x = x[0].to(device)
            y_pred = model(x)
            scores.append(y_pred.cpu().detach().numpy())
    scores = np.concatenate(scores, axis=0)
    return scores


def get_scores_Reptile(domainS_proj, domainT_proj, domainS_labels, n_lsamples, n_nfeat, num_steps=1000, inner_batch_size=5, inner_lr=0.1,
                       meta_lr=0.001):
    """使用 Reptile 算法获取查询集的得分"""
    # 将 numpy 矩阵转换为 PyTorch 张量
    domainS_proj = torch.from_numpy(domainS_proj).float()
    domainT_proj = torch.from_numpy(domainT_proj).float()
    domainS_labels = torch.from_numpy(domainS_labels).long()

    # 定义模型
    model = nn.Sequential(
        nn.Linear(n_nfeat, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 在支持集上训练模型
    for step in range(num_steps):
        if n_lsamples == 25:  # 5-shot
            # 从支持集中连续选择一个样本作为初始状态
            i = step % 5  # 循环选取0-4的样本，对应一个类别
            class_idx = (step // 5) % 5  # 循环选取5个类别
            start_idx = class_idx * 5 + i
            x_s = domainS_proj[start_idx:start_idx + 1]
            y_s = domainS_labels[start_idx:start_idx + 1]
        else:
            # 从支持集中随机选择一个样本作为初始状态
            i = np.random.randint(5)
            x_s = domainS_proj[i:i + 1]
            y_s = domainS_labels[i:i + 1]

        # 在支持集上执行几步梯度下降，更新模型参数
        for _ in range(inner_batch_size):
            # 前向传播
            y_pred = model(x_s)

            # 计算损失
            loss = criterion(y_pred, y_s)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印训练过程中的损失
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    # 在查询集上测试模型，获取得分
    scores = torch.zeros(75, 5)
    for i in range(5):
        if n_lsamples == 25:
            start_idx = i * 5
            x_s = domainS_proj[start_idx:start_idx + 5]
            y_s = domainS_labels[start_idx:start_idx + 5]
        else:
            x_s = domainS_proj[i:i + 1]
            y_s = domainS_labels[i:i + 1]

        # 在支持集上执行几步梯度下降，更新模型参数
        for _ in range(inner_batch_size):
            # 前向传播
            y_pred = model(x_s)

            # 计算损失
            loss = criterion(y_pred, y_s)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 在查询集上进行预测，计算得分
        x_t = domainT_proj[i * 15:i * 15 + 15]
        y_pred = model(x_t)
        scores[i * 15:i * 15 + 15] = y_pred.detach()

    # 将得分转换为 numpy 矩阵并返回
    return scores.numpy()


# ========================================
#      loading datas
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh


def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


def My_LPP(data, W, options):
    data = np.array(data, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    D = D.astype(np.float64)
    Sl = data.T @ L @ data
    Sd = data.T @ D @ data

    Sl = (Sl + Sl.T) / 2
    Sd = (Sd + Sd.T) / 2
    Sl = Sl + options["alpha"] * np.eye(Sl.shape[1])
    # 计算特征值和特征向量
    w, v = eigh(Sd, Sl)
    w, v = w[-options['ReducedDim']:], v[:, -options['ReducedDim']:]

    P = np.flip(v, axis=1)
    P = -P
    for i in range(P.shape[1]):
        if P[0, i] < 0:
            P[:, i] *= -1
    return P


def My_EuDist2(fea_a, fea_b, bSqrt=True):
    fea_a = np.array(fea_a)
    fea_b = np.array(fea_b)
    aa = np.sum(fea_a ** 2, axis=1)
    bb = np.sum(fea_b ** 2, axis=1)
    ab = fea_a @ fea_b.T

    aa = aa.reshape(-1, 1)
    bb = bb.reshape(-1, 1)

    D = np.add(aa, bb.T) - 2 * ab
    D[D < 0] = 0
    if bSqrt:
        D = np.sqrt(D)
    return D


def My_CosDist(fea_a, fea_b):
    fea_a = np.array(fea_a)
    fea_b = np.array(fea_b)
    dot_product = np.dot(fea_a, fea_b.T)
    norm_a = np.linalg.norm(fea_a, axis=1)
    norm_b = np.linalg.norm(fea_b, axis=1)
    return 1 - dot_product / (norm_a * norm_b)


def My_constructW(fea, options):
    nSmp = fea.shape[0]
    G = np.zeros((nSmp * (options['k'] + 1), 3))
    selected_fea = fea[0:options['WDim'], :]
    dist = My_EuDist2(selected_fea, fea, 0)
    # dist = My_CosDist(selected_fea, fea)
    nSmpNow = len(selected_fea)
    dump = np.zeros((nSmpNow, options['k'] + 1))
    idx = dump.copy()
    for j in range(options['k'] + 1):
        dump[:, j], idx[:, j] = np.min(dist, axis=1), np.argmin(dist, axis=1)
        temp = idx[:, j] * nSmpNow + np.arange(nSmpNow)  # python按行存储
        temp = temp.astype(int)
        temp = np.clip(temp, 0, dist.size - 1)
        for k in range(options['WDim']):
            index = temp[k]
            row, col = np.unravel_index(index,
                                        (options['WDim'], options['WDim']))  # 将索引i转换为对应的行、列下标，注意在Python中，行、列下标从0开始
            dist[col, row] = 1e100  # 将dist(row, col)赋值为1e100
    dump = np.exp(-dump / (2 * options['t'] ** 2))
    G[:, 0] = np.tile(np.arange(nSmp), options['k'] + 1)
    G[:, 1] = np.ravel(idx, order='F').flatten()
    G[:, 2] = np.ravel(dump, order='F').flatten()
    W = sparse.csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))
    W = W.toarray()
    W = W - np.diag(np.diag(W))
    W = np.maximum(W, W.transpose())
    return W


def LPP(ndatas, n_lsamples, options, W):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['ReducedDim']))
    supportX_2 = np.zeros((n_runs, n_lsamples, options['ReducedDim']))
    queryX_2 = np.zeros((n_runs, ndatas.shape[1] - n_lsamples, options['ReducedDim']))
    for i in range(n_runs):
        P[i] = My_LPP(np.concatenate((supportX[i], queryX[i])), W[i], options)
        # domainS_proj和domainT_proj，每轮将domainS_features和domainT_features降至5维
        domainS_proj = np.dot(supportX[i], P[i])
        domainT_proj = np.dot(queryX[i], P[i])
        proj_mean = np.mean(np.concatenate((domainS_proj, domainT_proj)), axis=0)
        domainS_proj = domainS_proj - np.tile(proj_mean, (domainS_proj.shape[0], 1))
        domainT_proj = domainT_proj - np.tile(proj_mean, (domainT_proj.shape[0], 1))
        domainS_proj = My_L2Norm(domainS_proj)
        domainT_proj = My_L2Norm(domainT_proj)
        supportX_2[i] = domainS_proj
        queryX_2[i] = domainT_proj
    ndatas = np.concatenate((supportX_2, queryX_2), axis=1)  # 在第一维上进行拼接
    ndatas = torch.from_numpy(ndatas)
    return ndatas


def get_LPP_datas(ndatas, n_lsamples, n_runs):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_sum = ndatas.shape[1]
    options = {'WDim': n_sum, 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'ReducedDim': 35,
               'alpha': 10}
    W = np.zeros((n_runs, n_sum, n_sum))
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas = LPP(ndatas, n_lsamples, options, W)  # 执行降维
    ndatas = ndatas.cuda()
    return ndatas


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]

    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries

        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test

        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)

        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):

        self.verbose = False
        self.progressBar = False
        self.alpha = alpha

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model, epochInfo=None):

        p_xj = model.getProbas()

        # average_probs = (p_xj[:, n_lsamples:, :] + 0.05 * probMatrix3) / 1.05
        # p_xj[:, n_lsamples:, :] = average_probs

        self.probas = p_xj

        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)

    def loop(self, model, n_epochs=20):

        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        # if self.progressBar:
        #     if type(self.progressBar) == bool:
        #         pb = tqdm(total = n_epochs)
        #     else:
        #         pb = self.progressBar

        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
            self.performEpoch(model, epochInfo=(epoch, n_epochs))

            # if (self.progressBar): pb.update()

        # get final accuracy and return it
        op_xj = model.getProbas()

        # 计算op_xj和probMatrix_all在概率维度上的平均值
        average_probs = (op_xj[:, n_lsamples:, :] + 0.6 * probMatrix3 + 0.3 * probMatrix1) / 2
        op_xj[:, n_lsamples:, :] = average_probs

        acc = self.getAccuracy(op_xj)
        return acc


if __name__ == '__main__':
    # ---- data loading
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniimagenet_both")  # iLPC的预训练模型和PT-MAP重复的是一样的
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    n_runs = FSLTask._maxRuns
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # Power transform
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub

    ndatas = centerDatas(ndatas)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    # 保存mat
    # from scipy.io import savemat
    # dataset = r"miniimagenetsf"
    # savemat(f'{dataset}_wrn_{n_shot}shots.mat', mdict={'features': ndatas.cpu().detach().numpy(), 'labels':labels.cpu().detach().numpy()})
    # print(111)

    start_time = time.time()  # 记录开始时间

    # LPP
    ndatas = get_LPP_datas(ndatas, n_lsamples, n_runs)

    n_nfeat = ndatas.size(2)

    # 4.对LPP降维后的数据，计算每个类的均值，再由均值与查询集的距离，获取probMatrix1，一个(75, 5)的概率距离，这部分可以替换
    supportX = ndatas[:, :n_lsamples, :].squeeze().cpu().numpy()
    queryX = ndatas[:, n_lsamples:, :].squeeze().cpu().numpy()
    supportY = labels[:, :n_lsamples].squeeze().cpu().numpy()

    probMatrix1 = np.zeros(((n_runs, n_usamples, n_ways)))
    probMatrix2 = np.zeros(((n_runs, n_usamples, n_ways)))
    probMatrix3 = np.zeros(((n_runs, n_usamples, n_ways)))
    for i in range(n_runs):
        domainS_proj = supportX[i]
        domainT_proj = queryX[i]
        domainS_labels = supportY[i]
        data = np.concatenate((domainS_proj, domainT_proj), axis=0)

        classMeans = np.zeros((n_ways, domainS_proj.shape[1]))
        for j in range(n_ways):
            classMeans[j, :] = np.mean(np.mean(domainS_proj[domainS_labels == j, :], axis=0))
        distClassMeans = My_EuDist2(domainT_proj, classMeans)  # 直接求距离 0.754

        # Reptile prob3,SVM prob1,Matching_Networks prob2 0.8818 0.8821 0.7955
        scores1 = get_scores_knn(domainS_proj, domainT_proj, domainS_labels)  # knn 0.8005
        # scores = get_scores_SVM(domainS_proj, domainT_proj, domainS_labels)  # SVM 0.8808 0.8891
        # scores = get_scores_meta_LSTM(domainS_proj, domainT_proj, domainS_labels, n_shot)  # meta 0.8648 慢
        # scores = get_scores_Matching_Networks(domainS_proj, domainT_proj, domainS_labels)  # Matching_Networks 0.8786 非深度学习网络
        scores2 = get_scores_ProtoNets(domainS_proj, domainT_proj, domainS_labels)  # ProtoNets 0.8781
        scores = get_scores_Deep_ProtoNets(domainS_proj, domainT_proj, domainS_labels, n_nfeat)  # 深度学习网络ProtoNets 0.7891(参数)
        # scores = get_scores_maml(domainS_proj, domainT_proj, domainS_labels)  # 0.5233
        # scores = get_scores_Reptile(domainS_proj, domainT_proj, domainS_labels, n_lsamples, n_nfeat)  # Reptile 0.8848 0.8882 较慢

        # expMatrix1 = np.exp(-distClassMeans)  # 直接求距离 0.754

        expMatrix1 = np.exp(scores1)
        expMatrix2 = np.exp(scores2)
        expMatrix3 = np.exp(scores)
        # expMatrix3 = scores

        probMatrix1[i] = expMatrix1 / np.repeat(np.sum(expMatrix1, axis=1).reshape(len(expMatrix1), 1), n_ways, axis=1)
        # _, probMatrix2[i] = Unsupervised_k_means(data, n_lsamples)  # k-means 0.817 不降维：0.877
        probMatrix2[i] = expMatrix2 / np.repeat(np.sum(expMatrix2, axis=1).reshape(len(expMatrix2), 1), n_ways, axis=1)
        probMatrix3[i] = expMatrix3 / np.repeat(np.sum(expMatrix3, axis=1).reshape(len(expMatrix3), 1), n_ways, axis=1)

        # probMatrix3 = probMatrix3 + 0.2 * (probMatrix2 + probMatrix1)
        # probMatrix[i] = probMatrix3

        # 5.sinkhorn算法调整probMatrix  My_sinkhornKnopp
        probMatrix1[i] = My_sinkhornKnopp(probMatrix1[i], 10, 1e-6)
        probMatrix2[i] = My_sinkhornKnopp(probMatrix2[i], 10, 1e-6)
        probMatrix3[i] = My_sinkhornKnopp(probMatrix3[i], 10, 1e-6)

    probMatrix1 = torch.from_numpy(probMatrix1).cuda()
    probMatrix2 = torch.from_numpy(probMatrix2).cuda()
    probMatrix3 = torch.from_numpy(probMatrix3).cuda()

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.2
    # alpha = 0.3
    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs=20)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")
    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))




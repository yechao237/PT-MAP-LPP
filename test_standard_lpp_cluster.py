import time
import torch
import math
import numpy as np
import scipy.sparse as sparse

from scipy.linalg import eigh
from scipy.io import savemat

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas


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
    w, v = w[-options['d']:], v[:, -options['d']:]

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


def LPP(ndatas, options, W):
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['d']))
    ndatas_2 = np.zeros((n_runs, ndatas.shape[1], options['d']))
    for i in range(n_runs):
        P[i] = My_LPP(ndatas[i], W[i], options)
        ndatas_proj = np.dot(ndatas[i], P[i])
        proj_mean = np.mean(ndatas_proj, axis=0)
        ndatas_proj = ndatas_proj - np.tile(proj_mean, (ndatas_proj.shape[0], 1))
        ndatas_proj = My_L2Norm(ndatas_proj)
        ndatas_2[i] = ndatas_proj
    ndatas = np.array(ndatas_2)
    return ndatas, P


def get_LPP_datas(ndatas):
    options = {'WDim': ndatas.shape[1], 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'd': 35,
               'alpha': 10}
    n_runs = ndatas.shape[0]
    ndatas = ndatas.cpu().numpy()  # 全部数据
    n_sum = ndatas.shape[1]  # 一个task中的数据个数
    W = np.zeros((n_runs, n_sum, n_sum))
    for i in range(n_runs):
        W[i] = My_constructW(ndatas[i, :, :].squeeze(), options)
    ndatas, P = LPP(ndatas, options, W)  # 执行降维
    ndatas = torch.from_numpy(ndatas).cuda()
    return ndatas, P


def centerDatas(datas, n_lsamples):
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

    def initFromLabelledDatas(self, mus=None):
        if mus == None:
            self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
        else:
            self.mus = mus

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
        # mask = torch.exp(mask)
        # mask = torch.sigmoid(mask)
        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None, verbose=False):

        self.verbose = verbose
        self.progressBar = verbose
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
        self.probas = p_xj

        # if self.verbose:
        #     print("accuracy from filtered probas", self.getAccuracy(self.probas))

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
        acc = self.getAccuracy(op_xj)
        return acc


def get_mus(support_datas, support_labels, n_nfeat):
    n_runs = support_datas.size(0)
    mus = torch.zeros((n_runs, 5, n_nfeat)).cuda()  # 初始化一个张量来存储计算出的均值
    for task_index in range(n_runs):  # 遍历每个任务
        task_data = support_datas[task_index]  # 当前任务的数据
        task_labels = support_labels[task_index]  # 当前任务的标签
        for class_index in range(5):  # 遍历每个类别
            class_indices = (task_labels == class_index).nonzero(as_tuple=True)[0]  # 找到当前类别的所有样本
            if len(class_indices) != 0:
                class_data = task_data[class_indices]  # 选择对应的数据
                mus[task_index, class_index] = class_data.mean(0)  # 计算当前类别的均值，并将其存储在mus张量中
            else:
                print(f"Warning：missing values is exist!")
                mus[task_index, class_index] = torch.zeros(n_nfeat).cuda()  # 如果不初始化为0 nan会使得无法进入sinkhorn中的迭代
                # mus[task_index, class_index] = task_data.mean(0)  # 初始化为这个task的均值
    return mus


from sklearn.cluster import KMeans


def cluster_data(support_datas, ndatas, cluster=5):
    # 获取任务数量
    num_tasks, _, feature_dim = support_datas.shape
    new_support_mus_datas = torch.empty_like(support_datas)
    for task in range(num_tasks):
        # 使用kmeans为ndatas的每个任务进行分类
        kmeans = KMeans(n_clusters=cluster).fit(ndatas[task].cpu().numpy())
        # 获取kmeans的聚类中心
        cluster_centers = torch.tensor(kmeans.cluster_centers_).cuda()
        for i in range(5):  # 遍历每个支持集的平均值
            distances = torch.norm(cluster_centers - support_datas[task, i, :], dim=1)
            closest_cluster = torch.argmin(distances).item()  # 获取距离最小的聚类中心索引
            new_support_mus_datas[task, i, :] = cluster_centers[closest_cluster, :]
    return new_support_mus_datas


def PT(ndatas, beta):
    nve_idx = np.where(ndatas.cpu().numpy() < 0)
    ndatas[nve_idx] *= -1
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas[nve_idx] *= -1  # return the sign
    return ndatas


def data_preprocessing(ndatas, labels, n_lsamples):
    beta = 0.5
    ndatas = PT(ndatas, beta)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas, n_lsamples)  # trans-mean-sub

    # switch to cuda
    ndatas = ndatas.cuda()  # fsl ndatas
    labels = labels.cuda()

    # LPP
    ndatas, _ = get_LPP_datas(ndatas)
    return ndatas, labels


if __name__ == '__main__':
    # 不适用AL
    # ---- data loading
    seed_value = 42  # 随机种子
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot  # n_lsamples表示已经标记的支持集，用于fsl
    n_usamples = n_ways * n_queries  # 75个查询集，用于fsl和afsl
    n_samples = n_lsamples + n_usamples  # 全部样本(已标记支持集+查询集)
    fsl_samples = n_lsamples + n_usamples  # 用于fsl训练的数据集

    import FSLTask

    verbose = True  # 是否输出高斯模型每轮的acc
    fsl = 2  # 0不进行fsl 1ptmaplpp 2ptmaplpp聚类
    n_clusters = n_lsamples  # 聚类的个数
    samples_per_cluster = int(n_lsamples / n_clusters)  # 聚类中选择样本的个数
    n_epochs = 8  # afsl中gaussian迭代轮数
    dataset = r"miniimagenet"
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}

    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    n_runs = FSLTask._maxRuns
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # step1: afsl——active get data: for every task, choose samples, and return active datas and labels
    ndatas, labels = data_preprocessing(ndatas, labels, n_lsamples)
    n_nfeat = ndatas.size(2)
    if fsl != 0:
        mus = None
        start_time = time.time()  # 记录开始时间
        if fsl == 2:
            # 对应的支持集
            # support_mus_datas = get_mus(ndatas, labels, n_lsamples, n_nfeat)
            support_datas = ndatas[:, :n_lsamples, :]
            support_labels = labels[:, :n_lsamples]

            # fsl 取聚类均值
            # ndatas[:, :n_lsamples, :] = cluster_data(support_datas, ndatas, cluster=n_lsamples)  # 这样改变了原本的支持集，结果变差
            support_datas = cluster_data(support_datas, ndatas, cluster=n_lsamples)
            mus = get_mus(support_datas, support_labels, n_nfeat)

        # MAP
        lam = 10
        model = GaussianModel(n_ways, lam)
        model.initFromLabelledDatas(mus)

        alpha = 0.3 if n_shot == 1 else 0.2
        optim = MAP(alpha)

        optim.verbose = False
        optim.progressBar = False

        acc_test = optim.loop(model, n_epochs=20)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算经过的时间
        print(f"fsl执行时间为: {elapsed_time} 秒")
        print("final fsl accuracy is {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))

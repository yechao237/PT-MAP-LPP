import time
import torch
import math
import numpy as np
import scipy.sparse as sparse

from scipy.linalg import eigh
from scipy.io import savemat
from sklearn.cluster import KMeans

torch.set_default_dtype(torch.float64)

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas

def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


def My_LPP(data, W, options):
    data = np.array(data, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    D = np.diag(np.sum(W, axis=1))  # formula 3
    L = D - W  # formula 4
    D = D.astype(np.float64)
    Sl = data.T @ L @ data  # formula 5
    Sd = data.T @ D @ data  # formula 6

    Sl = (Sl + Sl.T) / 2
    Sd = (Sd + Sd.T) / 2
    Sl = Sl + options["alpha"] * np.eye(Sl.shape[1])
    # 计算特征值和特征向量
    w, v = eigh(Sd, Sl)  # formula 7
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

    D = np.add(aa, bb.T) - 2 * ab  # formula 1
    D[D < 0] = 0
    if bSqrt:
        D = np.sqrt(D)
    return D


def My_constructW(fea, options):
    nSmp = fea.shape[0]
    G = np.zeros((nSmp * (options['k'] + 1), 3))
    selected_fea = fea[0:options['WDim'], :]
    dist = My_EuDist2(selected_fea, fea, 0)
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
    dump = np.exp(-dump / (2 * options['t'] ** 2))  # formula 2
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
    def __init__(self, n_ways, lam, ndatas, labels):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.ndatas = ndatas
        self.labels = labels

    def initFromLabelledDatas(self, mus=None):
        if mus == None:
            self.mus = self.ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
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


        # 按任务进行循环处理
        # for idx in range(n_runs):  # Loop over tasks
        #     iters = 1
        #     while torch.max(torch.abs(u[idx] - P[idx].sum(1))) > epsilon:
        #         u[idx] = P[idx].sum(1)
        #         P[idx] *= (r[idx] / u[idx]).view((-1, 1))
        #         P[idx] *= (c[idx] / P[idx].sum(0)).view((1, -1))
        #         if iters == maxiters:
        #             break
        #         iters = iters + 1

        # normalize this matrix
        iters = 1
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:  # 如果P当中存在nan，这里就不会执行了
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1

        # print(torch.max(torch.abs(u - P[88].sum(1))) > epsilon)  # miniimagenet nan
        return P, torch.sum(P * M)

    def getProbas(self):
        # self.mus[88, 3] = torch.zeros(35).cuda()
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (self.ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries
        p_xj = torch.zeros_like(dist)
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, self.labels[:, :n_lsamples].unsqueeze(2), 1)
        return p_xj

    def estimateFromMask(self, mask):
        emus = mask.permute(0, 2, 1).matmul(self.ndatas).div(mask.sum(dim=1).unsqueeze(2))
        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, labels, alpha=None):
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
        self.labels = labels

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = self.labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)
        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model):
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
        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
            self.performEpoch(model)
            # if (self.progressBar): pb.update()
        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc

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


def Gasussianloop(n_shot, n_queries, n_ways, ndatas, labels, n_epochs=20, verbose=False, mus=None):
    lam = 10
    alpha = 0.3 if n_shot == 1 else 0.2
    model = GaussianModel(n_ways, lam, ndatas, labels)
    model.initFromLabelledDatas(mus)
    optim = MAP(labels, alpha)
    optim.verbose = verbose
    optim.progressBar = verbose
    acc_test = optim.loop(model, n_epochs=n_epochs)
    return acc_test


def get_mus(active_ndatas, active_nlabels, n_lsamples, n_nfeat):
    n_runs = active_ndatas.size(0)
    support_datas = active_ndatas[:, :n_lsamples, :]
    support_labels = active_nlabels[:, :n_lsamples]
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

def cluster_data_and_labels(active_data, active_data_afsl, active_labels, dist=0, random=1, n_clusters=10, samples_per_cluster=2, random_state=42):
    # ** idea2 n_clusters * samples_per_cluster可以是25，50，100
    # ** idea3 n_clusters可以是5，10，20

    # 确保输入是正确的维度
    assert active_data.dim() == 3 and active_data_afsl.dim() == 3 and active_labels.dim() == 2, "Incorrect dimensions for data or labels."

    # 设置随机种子以确保可复制性
    np.random.seed(random_state)
    # 获取数据的尺寸
    num_tasks, num_samples, num_features = active_data_afsl.shape
    # 初始化输出张量
    clustered_data = torch.zeros((num_tasks, n_clusters * samples_per_cluster, num_features))
    clustered_labels = torch.zeros((num_tasks, n_clusters * samples_per_cluster), dtype=torch.int64)

    # 对于数据集中的每个任务，随机/AL
    for task_idx in range(num_tasks):
        # 从当前任务中提取数据和标签
        task_data = active_data[task_idx]
        task_labels = active_labels[task_idx]
        task_data_afsl = active_data_afsl[task_idx]

        if dist == 0 and random == 2:
            # 从全部样本中随机选择
            all_indices = np.arange(num_samples)
            selected_indices_global = np.random.choice(all_indices, n_clusters * samples_per_cluster, replace=False)
            for sample_idx, global_idx in enumerate(selected_indices_global):
                clustered_data[task_idx, sample_idx, :] = task_data_afsl[global_idx]
                clustered_labels[task_idx, sample_idx] = task_labels[global_idx]
            continue  # 跳过当前任务的后续代码，进入下一个任务的处理
        if dist == 0 and random == 3:
            # 基于真实标签，从每个类别中随机选择样本
            all_samples_selected = []
            for label in range(5):  # 标签是0, 1, 2, 3, 4
                # 找到当前类别的所有样本
                label_indices = (task_labels == label).nonzero(as_tuple=True)[0].cpu().numpy()
                # 从当前类别中随机选择样本  # replace=False 不重复抽样
                selected_indices = np.random.choice(label_indices, samples_per_cluster, replace=False)
                all_samples_selected.extend(selected_indices)
            # 从原始数据中提取所选样本
            for new_idx, original_idx in enumerate(all_samples_selected):
                clustered_data[task_idx, new_idx, :] = task_data_afsl[original_idx]
                clustered_labels[task_idx, new_idx] = task_labels[original_idx]
            continue  # 继续下一个任务的处理
        # 将数据转换为numpy数组，以适应scikit-learn的KMeans实现
        task_data_np = task_data.cpu().numpy()
        # 使用KMeans找到数据的簇
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(task_data_np)
        clusters = kmeans.labels_
        # 计算所有点到其相应簇中心的距离
        distances_to_center = np.sqrt(((task_data_np - kmeans.cluster_centers_[clusters]) ** 2).sum(axis=1))
        samples_selected = 0  # 记录已选择的样本数量
        # 从每个簇中提取样本
        for cluster_idx in range(n_clusters):
            # 获取当前簇的所有样本索引
            cluster_samples_idx = np.where(clusters == cluster_idx)[0]
            if dist == 0 and random == 1:
                if len(cluster_samples_idx) < samples_per_cluster:
                    # 当前簇中样本不够，重复抽样
                    selected_indices = np.random.choice(cluster_samples_idx, samples_per_cluster, replace=True)
                else:
                    # 从当前簇中随机选择样本
                    selected_indices = np.random.choice(cluster_samples_idx, samples_per_cluster, replace=False)
            else:
                # 计算当前簇中所有点到簇中心的距离
                cluster_distances = distances_to_center[cluster_samples_idx]
                if dist == 1:
                    # 找到距离的中位数
                    median_distance = np.median(cluster_distances)
                    # 找到最接近中位数的距离的点，可能只能选1个
                    target_indices = np.argsort(np.abs(cluster_distances - median_distance))[:samples_per_cluster]
                elif dist == 2:
                    # 找到距离最近的点，可能只能选1个
                    target_indices = np.argsort(cluster_distances)[:samples_per_cluster]
                elif dist == 3:
                    # 找到距离最远的点，可能只能选1个
                    target_indices = np.argsort(-cluster_distances)[:samples_per_cluster]  # 注意这里我们使用了负数排序来获得最远的点
                elif dist == 4:
                    # 按照距离排序，找到位于前t位置的样本 t = 0.1,0.2,...，可能只能选1个
                    t = 0.3
                    sorted_indices = np.argsort(cluster_distances)
                    t_percent_point = int(len(sorted_indices) * t)  # 计算前t的位置
                    t_percent_point = sorted_indices[t_percent_point] if t_percent_point < len(sorted_indices) else sorted_indices[-1]
                    # 计算与该点的距离L
                    L = cluster_distances[t_percent_point]
                    # 找到与簇中心距离最接近L的5个点
                    target_indices = np.argsort(np.abs(cluster_distances - L))[:samples_per_cluster]
                else:
                    raise ValueError("Invalid value for 'dist'. Choose among 0, 1, 2, 3, 4.")

                selected_indices = cluster_samples_idx[target_indices]
            # 获取选定样本的数据和标签
            selected_data = task_data_afsl[selected_indices]
            selected_labels = task_labels[selected_indices]
            # 将选定的样本存储到最终张量中
            start_idx = samples_selected
            if len(cluster_samples_idx) < samples_per_cluster:  # 当前簇中样本不够，循环赋值
                for i in range(samples_per_cluster):
                    data_idx = i % selected_data.shape[0]  # 这将在 0 到 num_selected_data-1 之间循环
                    clustered_data[task_idx, start_idx + i, :] = selected_data[data_idx]
                    clustered_labels[task_idx, start_idx + i] = selected_labels[data_idx]
            else:
                end_idx = start_idx + samples_per_cluster
                clustered_data[task_idx, start_idx:end_idx, :] = selected_data
                clustered_labels[task_idx, start_idx:end_idx] = selected_labels
            samples_selected += samples_per_cluster
    return clustered_data, clustered_labels


def compute_optimal_transport(M, r, c, epsilon=1e-6):
    lam = 10
    r = r.cuda()
    c = c.cuda()
    n_runs, n, m = M.shape
    P = torch.exp(- lam * M)
    P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000

    # normalize this matrix
    iters = 1
    while torch.max(torch.abs(u - P.sum(2))) > epsilon:  # 如果P当中存在nan，这里就不会执行了
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        if iters == maxiters:
            break
        iters = iters + 1

    # print(torch.max(torch.abs(u - P[88].sum(1))) > epsilon)  # miniimagenet nan
    return P, torch.sum(P * M)


def cluster_data_and_labels_sinkhorn(active_data, active_data_afsl, active_label):
    """
    :param active_data: A CUDA tensor of shape (num_tasks, num_samples, num_features)
                        representing the data for multiple tasks.
    :param active_data_afsl: A CUDA tensor of the same shape as active_data, representing
                             the data from which we will select the support set.
    :param active_label: A CUDA tensor of shape (num_tasks, num_samples) with int64 elements,
                         representing the labels of the data.
    :return: Two tensors, support_datas and support_labels, each of shape (num_tasks, 25, [num_features or 1]),
             representing the selected support data and labels.
    """
    num_tasks, num_samples, num_features = active_data.shape
    num_clusters = 5
    n_queries = num_samples // num_clusters  # Ensure this is integer division

    dist = torch.zeros((num_tasks, num_samples, num_clusters)).cuda()
    for i in range(num_tasks):
        # Extract the data for the current task
        task_data = active_data[i].cpu().numpy()
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit(task_data)
        cluster_centers = torch.tensor(kmeans.cluster_centers_).cuda()
        # Compute the distance from each sample to each cluster center
        for j in range(num_clusters):
            dist[i, :, j] = torch.norm(active_data[i] - cluster_centers[j], dim=1)

    # sinkhorn
    r = torch.ones(num_tasks, num_samples).cuda()
    c = torch.ones(num_tasks, num_clusters).cuda() * n_queries
    p_xj, _ = compute_optimal_transport(dist, r, c, epsilon=1e-6)
    entropy = -torch.sum(p_xj * torch.log(p_xj + 1e-9), dim=2)  # 计算熵，加入1e-9避免log(0)的情况

    # Selecting the samples with the minimum entropy within each cluster
    support_datas = torch.zeros((num_tasks, num_clusters * 5, num_features)).cuda()
    support_labels = torch.zeros((num_tasks, num_clusters * 5)).long().cuda()  # Assuming labels are integers

    for i in range(num_tasks):
        cluster_indices = torch.argmin(dist[i], dim=1)  # Indices of the closest cluster for each sample
        for j in range(num_clusters):
            # Find the indices of the samples that belong to cluster j
            in_cluster_idx = torch.nonzero(cluster_indices == j, as_tuple=False).squeeze(-1)
            cluster_entropy = entropy[i, in_cluster_idx]

            # Find the indices of the 5 samples with the lowest entropy
            _, low_entropy_idx = torch.topk(cluster_entropy, 5, largest=False)

            selected_samples = in_cluster_idx[low_entropy_idx]

            support_datas[i, j * 5:(j + 1) * 5] = active_data_afsl[i, selected_samples]
            support_labels[i, j * 5:(j + 1) * 5] = active_label[i, selected_samples]

    return support_datas, support_labels

def check_values(tensor, values):
    for i in range(tensor.size(0)):
        unique_values = set(torch.unique(tensor[i]).cpu().numpy())
        # 检查目标值是否都在唯一值集合中
        if values.issubset(unique_values):
            continue
        else:
            missing_values = values - unique_values
            print(f"Warning：missing_values：{missing_values}")


if __name__ == '__main__':
    # ---- data loading
    seed_value = 42  # 随机种子
    n_shot = 1
    n_ways = 5
    n_queries = 15
    # n_unlabelled miniimagenet、cifar最多580，tieredimagenet最多934，cub最多28
    n_unlabelled = 100
    n_lsamples = n_ways * n_shot  # n_lsamples表示已经标记的支持集，用于fsl
    n_usamples = n_ways * n_queries  # 75个查询集，用于fsl和afsl
    active_samples = n_ways * n_unlabelled  # 未标记的支持集 500/140个未标记的支持集(cub 140)   ** idea1:不均匀的情况
    n_samples = n_lsamples + n_usamples + active_samples  # 全部样本(已标记支持集+未标记支持集+查询集)
    fsl_samples = n_lsamples + n_usamples  # 用于fsl训练的数据集

    import FSLTask
    fsl = 1  # 是否进行fsl，为1进行fsl，为0不进行
    balanced = True  # 是否均匀抽取
    dist_type = 2  # 0随机 1中位数距离 2最小距离 3最大距离 4距离比例
    random_type = 1  # dist_type==0时， 1聚类后随机 2全部随机 3按照真实标签随机
    n_clusters = 5  # 聚类的个数
    samples_per_cluster = int(n_shot * n_ways / n_clusters)  # 聚类中选择样本的个数
    n_epochs = 20  # gaussian迭代轮数
    dataset = r"cifar"
    # 均匀抽取的设置1
    if balanced == True:
        cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries + n_unlabelled}  # n-shot 5-way 115 查询集+未标记支持集
    # 不均匀抽取的设置1
    else:
        n_unlabelled_select = 500
        n_samples_select = n_lsamples + n_usamples + n_ways * n_unlabelled_select
        cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries + n_unlabelled_select}  # n-shot 5-way 515 查询集+未标记支持集

    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    n_runs = FSLTask._maxRuns
    all_ndatas = FSLTask.GenerateRunSet(cfg=cfg)

    # 均匀抽取的设置2
    if balanced == True:
        all_ndatas = all_ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries + n_unlabelled, 5).clone().view(n_runs,
                                                                                                            n_samples)
    # 不均匀抽取的设置2
    else:
        all_ndatas = all_ndatas.permute(0, 2, 1, 3)
        all_ndatas = all_ndatas.numpy()
        all_ndatas = all_ndatas.reshape(n_runs, n_samples_select, -1)
        print(all_ndatas.shape)
        labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries + n_unlabelled_select, 5).clone().view(
            n_runs, n_samples_select)
        # 设置随机数种子以确保结果的可重复性
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        # 选择随机索引
        num_random_samples = 500
        total_samples = fsl_samples + num_random_samples
        # 从范围 [100, 2600) 中选择 500 个随机索引
        random_indices = np.random.choice(np.arange(fsl_samples, n_samples_select), num_random_samples, replace=False)
        # 合并这些索引与前 100 个样本的索引
        selected_indices = np.concatenate([np.arange(fsl_samples), random_indices])
        # 根据选择的索引提取数据
        all_ndatas = all_ndatas[:, selected_indices, :]
        labels = labels[:, selected_indices]
        all_ndatas = torch.from_numpy(all_ndatas)
        print(all_ndatas.shape, labels.shape)
        # 清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # partition for afsl learning
    ndatas, active_data = all_ndatas[:, :fsl_samples, :], all_ndatas[:, fsl_samples:, :]  # fsl的数据和未标记支持集数据
    labels, active_label = labels[:, :fsl_samples], labels[:, fsl_samples:]  # fsl的标签和未标记支持集标签
    active_ndatas = ndatas[:, n_lsamples:, :].clone()  # afsl的初始查询集数据
    active_nlabels = labels[:, n_lsamples:]  # afsl的初始查询集标签
    print(ndatas.shape, active_ndatas.shape, active_data.shape, labels.shape, active_nlabels.shape, active_label.shape)

    # step1: fsl
    # Power transform
    if fsl == 1:
        ndatas, labels = data_preprocessing(ndatas, labels, n_lsamples)
        n_nfeat = ndatas.size(2)
        # MAP
        acc_test = Gasussianloop(n_shot, n_queries, n_ways, ndatas, labels)
        print("fsl final accuracy with 15 queries: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))

    # step2: afsl  get data: for every task, choose 25 samples, and return active datas and labels
    active_data_afsl = active_data.clone()
    active_data, active_label = data_preprocessing(active_data, active_label, n_lsamples)
    print(active_data.shape, active_label.shape)

    start_time = time.time()  # 记录开始时间
    # dist=0 and random=1,2,3 表示随机选 1:按类随机5*5  2:全部随机25  3:按真实标签随机5*5(相当于5-shot fsl)
    # dist=1/2/3表示根据dist选，为afsl 根据类均值的距离远近从每个聚类中选  1:距离中位数的5个  2:距离最小的5个  3:距离最大的5个  4:根据距离的比例选

    # 通过计算距离，当聚类5时结果不如随机，聚类10结果好于随机，一般聚类越多，AL结果越好
    support_datas, support_labels = cluster_data_and_labels(active_data, active_data_afsl, active_label, dist=dist_type, random=random_type,
                                                            n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, random_state=seed_value)
    # 聚类之后，通过距离计算最优传输，得到prob，再计算熵，根据熵挑选，结果没有变好
    # support_datas, support_labels = cluster_data_and_labels_sinkhorn(active_data, active_data_afsl, active_label)

    active_ndatas = torch.cat([support_datas.cpu(), active_ndatas], dim=1)
    active_nlabels = torch.cat([support_labels.cpu(), active_nlabels], dim=1)

    # target_values = {0, 1, 2, 3, 4}
    # check_values(support_labels, target_values)

    # step3: afsl
    active_ndatas, active_nlabels = data_preprocessing(active_ndatas, active_nlabels, n_lsamples)  # 数据预处理
    n_nfeat = active_ndatas.size(2)

    mus = get_mus(active_ndatas, active_nlabels, n_lsamples, n_nfeat)  # 获取支持集均值

    acc_test = Gasussianloop(n_shot, n_queries, n_ways, active_ndatas, active_nlabels, n_epochs=n_epochs, verbose=True, mus=mus)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")
    print("afsl accuracy2: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))

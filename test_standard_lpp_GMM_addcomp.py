import numpy as np
import torch
import math
import time

from scipy import sparse
from scipy.linalg import eigh

use_gpu = torch.cuda.is_available()

# ========================================
#      loading datas

def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    return datas

def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms

def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas

def SVDreduction(ndatas,K):
    # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
    # ndatas = ndatas.permute(0, 2, 1)
    _,s,v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:,:,:K])
    return ndatas

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianMixupModel(Model):
    def __init__(self, n_ways, lam, k, component):
        super(GaussianMixupModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][component][n_nfeat]
        self.sigmas = None  # shape [n_runs][n_ways][component][n_nfeat][n_nfeat]
        self.pis = None  # shape [n_runs][n_ways][component]
        self.lam = lam
        self.k = k
        self.component = component

    def initFromLabelledDatas(self):
        # 初始化均值、协方差矩阵和混合系数
        self.mus = torch.zeros((n_runs, n_ways, self.component, n_nfeat), dtype=torch.float64, device=ndatas.device)
        self.sigmas = torch.eye(n_nfeat, dtype=torch.float64, device=ndatas.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n_runs, n_ways, self.component, 1, 1)
        self.pis = torch.full((n_runs, n_ways, self.component), 1.0 / self.component, dtype=torch.float64, device=ndatas.device)

        if self.component == 1:
            # 对于单个组件，直接计算均值
            self.mus = ndatas.view(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(dim=1).cuda()
        else:
            # 使用K-means初始化均值
            from sklearn.cluster import KMeans
            reshaped_data = ndatas.reshape(n_runs, n_shot+n_queries, n_ways, n_nfeat)
            for run in range(n_runs):
                for way in range(n_ways):
                    samples = reshaped_data[run, :n_shot, way].cpu().numpy()
                    kmeans = KMeans(n_clusters=self.component, random_state=0, n_init=10).fit(samples)
                    self.mus[run, way] = torch.tensor(kmeans.cluster_centers_).to(ndatas.device)

    def updateFromEstimate(self, epis, emus, esigmas, alpha):
        self.mus = (1 - alpha) * self.mus + alpha * emus
        self.sigmas = (1 - alpha) * self.sigmas + alpha * esigmas
        self.pis = (1 - alpha) * self.pis + alpha * epis

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        # 转换为PyTorch tensor并分配给正确的设备，同时确保dtype正确
        M = M.to(dtype=torch.float64, device='cuda')
        r = r.to(dtype=torch.float64, device='cuda')
        c = c.to(dtype=torch.float64, device='cuda')
        n_runs, n, m = M.shape
        # 计算初始的P值
        P = torch.exp(-self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
        u = torch.zeros(n_runs, n, device="cuda", dtype=torch.float64)
        maxiters = 1000
        iters = 1
        # 迭代法规范化矩阵
        while torch.max(torch.abs(u - P.sum(dim=2))) > epsilon:
            u = P.sum(dim=2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(dim=1)).view((n_runs, 1, -1))
            if iters == maxiters:
                print("Reached maximum iterations. Proceeding with the current result.")
                break
            iters += 1
        # 计算最终的乘积
        temp = torch.sum(P * M)
        return P, temp

    def multivariate_gaussian_pdf(self, x, mean, cov):
        k = mean.shape[1]
        # 我们需要确保批处理维度的存在，因此我们增加一个维度来计算xm
        xm = x - mean
        # 首先，我们尝试在循环外预计算协方差矩阵的行列式和逆
        det = torch.linalg.det(cov)  # [batch_size]
        inv_cov = torch.linalg.inv(cov)  # [batch_size, k, k]
        # 对于正规化因子，由于除以的是批次中每个矩阵的行列式，我们可以使用批处理操作来计算所有的行列式
        normalization_factor = 1.0 / (torch.sqrt((2 * torch.pi) ** k * det))
        # 计算指数部分，我们也需要调整代码来一次处理整个批次
        # xm: [batch_size, k, 1] -> 使用broadcasting使其与逆协方差矩阵兼容
        # inv_cov: [batch_size, k, k]
        xm_expanded = xm.unsqueeze(-1)  # [batch_size, k, 1]
        exponent = -0.5 * torch.matmul(torch.matmul(xm_expanded.transpose(-2, -1), inv_cov), xm_expanded).squeeze(
            -1).squeeze(-1)  # [batch_size]
        # 最后，使用批处理操作计算最终的概率
        prob = normalization_factor * torch.exp(exponent)  # [batch_size]
        return prob

    def getProbas(self):
        device = ndatas.device
        batch_size, n_samples, n_nfeat = ndatas.shape
        # 初始化高斯概率张量
        gaussian_probs = torch.zeros((ndatas.shape[0], ndatas.shape[1], self.n_ways, self.component),
                                     dtype=torch.float64, device='cuda')

        for i in range(self.n_ways):
            for j in range(n_samples):
                # 为了消除 comp 循环，我们需要批量处理所有的 components
                # 我们可以通过增加额外的维度并使用广播来实现这一点

                # 将数据扩展到 component 维度
                data_sample = ndatas[:, j, :].unsqueeze(1)  # [batch_size, 1, k]

                # 获取当前所有 components 的均值和协方差
                current_mus = self.mus[:, i, :, :]  # [batch_size, self.component, k]
                current_sigmas = self.sigmas[:, i, :, :, :]  # [batch_size, self.component, k, k]

                # 由于 multivariate_gaussian_pdf 现在将同时处理所有 components，
                # 我们需要调整它以支持多个 component 的计算
                probs = self.multivariate_gaussian_pdf(data_sample, current_mus,
                                                       current_sigmas)  # [batch_size, self.component]

                # 同样地，我们需要从 pis 中获取所有相关的 component 数据
                pis = self.pis[:, i, :]  # [batch_size, self.component]

                # 现在，我们不需要 comp 循环，可以直接计算所有 components 的高斯概率
                gaussian_probs[:, j, i, :] = pis * probs

        # 在组件维度上合并概率，得到每个类别的总概率
        gaussian_probs = gaussian_probs.sum(dim=3)
        # 标准化概率
        gaussian_probs = gaussian_probs / gaussian_probs.sum(dim=2, keepdim=True)
        # 初始化概率矩阵 p_xj
        p_xj = torch.zeros_like(gaussian_probs)
        # 计算sinkhorn距离
        gaussian_distances = -torch.log(gaussian_probs + 1e-9)  # 为了避免对0取对数
        r = torch.ones((ndatas.shape[0], n_samples - n_lsamples), device=device, dtype=torch.float64)
        c = torch.ones((ndatas.shape[0], n_ways), device=device, dtype=torch.float64) * n_queries
        p_xj_test, _ = self.compute_optimal_transport(gaussian_distances[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:, :n_lsamples] = 0
        for i in range(n_lsamples):
            p_xj[range(batch_size), i, labels[:, i].long()] = 1
        return p_xj

    def m_step(self, X, weights, epsilon=1e-6):
        # 假设X和weights已经是在GPU上的tensors（如果你使用CUDA的话）
        batch_size, T, n_features = X.shape
        # 使用适当的tensor初始化方法而不是np.zeros
        new_mus = torch.zeros((batch_size, self.k, n_features), device=X.device, dtype=torch.float64)
        new_sigmas = torch.zeros((batch_size, self.k, n_features, n_features), device=X.device, dtype=torch.float64)
        # 遍历每一个批次
        for b in range(batch_size):
            X_batch = X[b]
            weights_batch = weights[b]
            # 遍历每一个高斯分布
            for i in range(self.k):
                # 获取第i个高斯分布对应的权重
                weight_i = weights_batch[:, i]
                # 计算权重的总和
                total_weight = weight_i.sum()
                # 防止除以0
                total_weight = torch.clamp(total_weight, min=epsilon)
                # 更新第i个高斯分布的均值
                new_mus[b, i] = (X_batch * weight_i[:, None]).sum(axis=0) / total_weight
                # 更新第i个高斯分布的协方差矩阵
                diff = (X_batch - new_mus[b, i])
                new_sigmas[b, i] = (diff.T @ (diff * weight_i[:, None])) / total_weight + epsilon * torch.eye(
                    n_features, device=X.device)
        return new_mus, new_sigmas

    def estimateFromMask(self, mask):
        # 'mask' 的维度是 [batch_size, n_samples, n_ways]
        # 我们需要为每个组件创建一个新的mask
        extended_mask = mask.unsqueeze(3).expand(-1, -1, -1,
                                                 self.component)  # 新维度为 [batch_size, n_samples, n_ways, component]
        # 初始化存储每个组件的期望参数的变量
        epis = torch.zeros((mask.shape[0], self.n_ways, self.component), device=mask.device, dtype=torch.float64)
        emus = torch.zeros((mask.shape[0], self.n_ways, self.component, ndatas.shape[2]), device=mask.device,
                           dtype=torch.float64)
        esigmas = torch.zeros((mask.shape[0], self.n_ways, self.component, ndatas.shape[2], ndatas.shape[2]),
                              device=mask.device, dtype=torch.float64)
        # 对于每个组件，我们需要计算期望参数
        for comp in range(self.component):
            # 从扩展的mask中提取特定组件的mask
            component_mask = extended_mask[:, :, :, comp]
            # 计算这个组件的期望参数
            epis[:, :, comp] = component_mask.mean(dim=1)
            # 对于均值和协方差，我们需要更复杂的计算，这通常在m步骤中完成
            # 我们传入特定组件的mask来计算这些参数
            emus_comp, esigmas_comp = self.m_step(ndatas, component_mask)
            # 将计算出的参数存储到相应的变量中
            emus[:, :, comp, :] = emus_comp
            esigmas[:, :, comp, :, :] = esigmas_comp
        return epis, emus, esigmas
          
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
        acc_test = matches[:,n_lsamples:].mean(1)
        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
        p_xj = model.getProbas()
        self.probas = p_xj
        self.verbose = True
        # if self.verbose:
        #     print("accuracy from filtered probas", self.getAccuracy(self.probas))
        epis, emus, esigmas = model.estimateFromMask(self.probas)
        # update centroids
        model.updateFromEstimate(epis, emus, esigmas, self.alpha)
        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, n_epochs=20):
        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            # if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc


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
            row, col = np.unravel_index(index, (options['WDim'], options['WDim']))  # 将索引i转换为对应的行、列下标，注意在Python中，行、列下标从0开始
            import warnings
            # Suppress all RuntimeWarnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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


def get_LPP_datas(ndatas, n_lsamples, n_runs):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_sum = ndatas.shape[1]
    options = {'WDim': n_sum, 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'd': 35, 'alpha': 10}
    W = np.zeros((n_runs, n_sum, n_sum))
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas = LPP(ndatas, n_lsamples, options, W)  # 执行降维
    ndatas = ndatas.cuda()
    return ndatas


def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


def LPP(ndatas, n_lsamples, options, W):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['d']))
    supportX_2 = np.zeros((n_runs, n_lsamples, options['d']))
    queryX_2 = np.zeros((n_runs, ndatas.shape[1] - n_lsamples, options['d']))
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


if __name__ == '__main__':
    # ---- data loading
    n_shot = 6
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    n_runs = FSLTask._maxRuns
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet("miniimagenet")  # iLPC的预训练模型和PT-MAP重复的是一样的
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    
    # Power transform
    beta = 0.5
    ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)

    # ndatas = SVDreduction(ndatas,40)
    ndatas = QRreduction(ndatas)

    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub
   
    ndatas = centerDatas(ndatas)
    
    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    ndatas = get_LPP_datas(ndatas, n_lsamples, n_runs)
    n_nfeat = ndatas.size(2)

    # MAP
    lam = 10
    k = 5
    model = GaussianMixupModel(n_ways, lam, k, component=6)
    model.initFromLabelledDatas()
    
    alpha = 0.2
    # alpha = 0.3
    optim = MAP(alpha)

    optim.verbose=True
    optim.progressBar=True
    start_time = time.time()  # 记录开始时间
    acc_test = optim.loop(model, n_epochs=20)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")

    print("final accuracy found {:0.2f}±{:0.2f}".format(*(100*x for x in acc_test)))

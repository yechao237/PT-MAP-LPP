import torch
import math
import time

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
    def __init__(self, n_ways, lam, k):
        super(GaussianMixupModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.k = k

    def initFromLabelledDatas(self):
        self.mus = ndatas.view(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(dim=1).cuda()
        self.sigmas = torch.eye(n_nfeat).unsqueeze(0).repeat(n_runs, n_ways, 1, 1).cuda()
        self.pis = torch.full((n_runs, n_ways), 1.0 / n_ways).cuda()

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
        # 确保输入是双精度tensor，并且已经被移动到GPU
        x = x.to(dtype=torch.float64, device='cuda')
        mean = mean.to(dtype=torch.float64, device='cuda')
        cov = cov.to(dtype=torch.float64, device='cuda')

        device = torch.device("cuda")  # 指定设备为 "cuda"

        batch_size = x.shape[0]
        k = mean.shape[1]

        # 我们需要确保批处理维度的存在，因此我们增加一个维度来计算xm
        xm = x - mean

        # 初始化概率数组
        prob = torch.zeros((batch_size,), device=device, dtype=torch.float64)

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

        # 初始化高斯概率矩阵
        gaussian_probs = torch.zeros((batch_size, n_samples, self.n_ways), device=device, dtype=torch.float64)

        # 计算数据点与每个高斯分布的概率
        for i in range(n_ways):
            for j in range(n_samples):
                gaussian_probs[:, j, i] = self.pis[:, i] * self.multivariate_gaussian_pdf(ndatas[:, j, :], self.mus[:, i, :],
                                                                                self.sigmas[:, i, :, :])

        # Normalize the probabilities
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
        epis = mask.mean(dim=1)
        emus, esigmas = self.m_step(ndatas, mask)
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


if __name__ == '__main__':
    # ---- data loading
    n_shot = 7
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    
    import FSLTask
    n_runs = FSLTask._maxRuns
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet("miniimagenet")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    
    # Power transform
    beta = 0.5
    ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)

    # ndatas = SVDreduction(ndatas,40)
    ndatas = QRreduction(ndatas)
    n_nfeat = ndatas.size(2)
    
    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub
   
    ndatas = centerDatas(ndatas)
    
    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()
    
    # MAP
    lam = 10
    k = 5
    model = GaussianMixupModel(n_ways, lam, k)
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

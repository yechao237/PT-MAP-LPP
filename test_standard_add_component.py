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
class GaussianModel(Model):
    def __init__(self, n_ways, lam, component=2):
        super(GaussianModel, self).__init__(n_ways)
        self.component = component
        self.mus = torch.zeros((n_runs, n_ways, component, n_nfeat)).cuda()  # shape [n_runs][n_ways][component][n_nfeat]
        self.lam = lam

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()



    def initFromLabelledDatas(self):
        from sklearn.cluster import KMeans
        reshaped_data = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)
        for run in range(n_runs):
            for way in range(n_ways):
                # 获取该类别的前n_shot个样本
                samples = reshaped_data[run, :n_shot, way].cpu().numpy()

                # 如果只有一个高斯分布，直接计算均值
                if self.component == 1:
                    self.mus[run, way, 0] = torch.from_numpy(samples.mean(axis=0)).cuda()
                else:
                    # 使用K-means聚类来初始化均值
                    kmeans = KMeans(n_clusters=self.component, random_state=0).fit(samples)
                    self.mus[run, way, :] = torch.tensor(kmeans.cluster_centers_)

    def updateFromEstimate(self, estimate, alpha):
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * Dmus
        # print(torch.sum(self.mus), torch.sum(estimate))

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

    # class GaussianModel(Model):
    #     def __init__(self, n_ways, lam, component=1):
    #         super(GaussianModel, self).__init__(n_ways)
    #         self.component = component
    #         self.mus = torch.zeros((n_runs, n_ways, component, n_nfeat))  # shape [n_runs][n_ways][component][n_nfeat]
    #         self.lam = lam

    def getProbas(self):
        all_dists = []
        for j in range(self.component):
            dist = (ndatas.unsqueeze(2) - self.mus[:, :, j].unsqueeze(1)).norm(dim=3).pow(2)
            all_dists.append(dist)

        if self.component > 1:
            all_dists = torch.stack(all_dists, dim=3)  # shape [n_runs][n_samples][n_ways][component]
            dist, _ = all_dists.min(dim=3)  # 选择最小的距离
        else:
            dist = all_dists[0]

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries

        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)

        p_xj[:, n_lsamples:] = p_xj_test

        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)

        return p_xj

    def estimateFromMask(self, mask):
        emus = torch.zeros_like(self.mus)
        for j in range(self.component):
            emus[:, :, j] = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
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
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
        p_xj = model.getProbas()
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
    n_shot = 5
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
    model = GaussianModel(n_ways, lam, component = 2)
    model.initFromLabelledDatas()
    
    # alpha = 0.2
    alpha = 0.3
    optim = MAP(alpha)

    optim.verbose=True
    optim.progressBar=True
    start_time = time.time()  # 记录开始时间
    acc_test = optim.loop(model, n_epochs=20)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")

    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
    
    


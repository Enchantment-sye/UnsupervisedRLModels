import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIsolationKernel(nn.Module):
    """
    High-efficiency GPU version of Soft-assignment Isolation Kernel (iNNE).
    """

    def __init__(self, input_dim, ensemble_size=100, subsample_size=32, temperature=0.01, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        self.temperature = temperature
        self.device = device

        # 一个大 anchor 矩阵 (ensemble_size * subsample_size, input_dim)
        total_anchors = ensemble_size * subsample_size
        self.anchors = nn.Parameter(torch.randn(total_anchors, input_dim, device=device),
                                    requires_grad=False)

    @torch.no_grad()
    def fit(self, data):
        """
        从数据池 (N, D) 中随机采样，更新所有 ensemble 的 anchors。
        """
        N = data.shape[0]
        total_anchors = self.ensemble_size * self.subsample_size
        idx = torch.randint(0, N, (total_anchors,), device=data.device)
        self.anchors.data = data[idx].clone().to(self.device)

    def kernel_mean(self, data, groups: int = 1):
        """
        计算组表征
        """
        if groups < 1:
            raise ValueError('groups must be >= 1')
        if groups == 1:
            return torch.mean(self.forward(data), dim=0)
        group_datas = torch.chunk(data, groups, dim=0)

        kernel_means = []
        for group_data in group_datas:
            kernel_means.append(torch.mean(self.forward(group_data), dim=0))
        if data.shape[0] % groups == 0:
            return torch.mean(torch.stack(kernel_means), dim=0)
        batch_size, remain_size = len(group_datas[0]), len(group_datas[-1])

        return ( batch_size * (groups - 1) * torch.mean(torch.stack(kernel_means[:groups - 1]), dim=0) + remain_size * kernel_means[-1] ) / data.shape[0]

    def forward(self, x):
        """
        输入: x (batch, D)
        输出: features (batch, ensemble_size * subsample_size)
        """

        features = self.compute_ik_map(x)
        return features

    def compute_ik_map(self, x):
        B, D = x.shape
        total_anchors = self.ensemble_size * self.subsample_size

        # pairwise distances: (B, total_anchors)
        dist = torch.cdist(x, self.anchors, p=2)  # GPU batch 计算

        # reshape -> (B, ensemble_size, subsample_size)
        dist = dist.view(B, self.ensemble_size, self.subsample_size)

        # softmax over subsample dimension
        assign = F.softmax(-dist / self.temperature, dim=-1)

        # flatten back -> (B, ensemble_size * subsample_size)
        features = assign.reshape(B, total_anchors)
        return features

# batch of encoded states
# batch_size, d = 1024, 64
# x = torch.randn(batch_size, d, device="cuda", requires_grad=True)
#
# # 构建 kernel
# ikernel = SoftIsolationKernel(input_dim=d,
#                               ensemble_size=200,
#                               subsample_size=16,
#                               temperature=0.05,
#                               device="cuda")
#
#
#
# # 初始化 anchors (来自 replay buffer)
# data_pool = torch.randn(50000, d, device="cuda")
# ikernel.set_anchors(data_pool)
# breakpoint()
# # 前向计算 (GPU 上并行)
# features = ikernel(x)  # (1024, 200*16)
# print(features.shape)   # torch.Size([1024, 3200])
#

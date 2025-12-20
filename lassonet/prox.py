import torch
from torch.nn import functional as F 

# https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py#L60

def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def prox(v, u, *, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    
    Tài liệu hóa: Mapping với Algorithm 4 trong bài báo LassoNet (Group-Hier-Prox)
    Input:
    - v: Tương ứng với theta (trọng số lớp skip connection)
    - u: Tương ứng với W^(1) (trọng số lớp ẩn đầu tiên)
    - lambda_, M: siêu tham số

    Giải thích sự tương thích với Algorithm 2 (Hier-Prox):
    Algorithm 2 là trường hợp đặc biệt của Algorithm 4 khi số chiều của theta là m=1 (scalar).
    
    1. Input:
       - Trong Alg 2, theta là scalar. Trong code này, nếu input v là 1D (onedim=True), nó được unsqueeze thành vector (1, batch).
    
    2. Sort (Line 3):
       - Việc sắp xếp W (u) là giống hệt nhau trong cả 2 thuật toán.
       
    3. Công thức cập nhật w_m (Line 5):
       - Alg 2 dùng |theta|. Code dùng ||v||_2 (norm 2).
       - Với vector 1 chiều v, ||v||_2 = |v|. Do đó công thức tính toán là tương đương.
       
    4. Cập nhật theta_tilde (Line 8 Alg 2 vs Line 9 Alg 4):
       - Alg 4: theta <- (1/M) * w * (theta / ||theta||_2)
       - Alg 2: theta <- (1/M) * w * sign(theta)
       - Với theta scalar: theta / ||theta||_2 = theta / |theta| = sign(theta).
       - Vậy công thức cập nhật trùng khớp.
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    # Alg 4, Line 4: Sắp xếp các tọa độ của W^(1) (u) theo giá trị tuyệt đối giảm dần.
    # |W_(j,1)| >= ... >= |W_(j,K)|
    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    # Alg 4, Line 5-6: Tính toán các ứng viên w_{j,m} cho m = 0...K
    # Công thức: w_{j,m} = M / (1 + m * M^2) * SoftThreshold(...)
    # Code thực hiện tính toán vector hóa cho tất cả m cùng lúc.
    
    # Tính tổng tích lũy (prefix sum) của |W|
    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    # Biến x tương ứng với thành phần chưa nhân M và norm_v trong công thức w_{j,m}
    x = F.relu(1 - a_s / norm_v) / (1 + s * M**2)

    # w ở đây chính là w_{j,m} trong thuật toán
    w = M * x * norm_v
    
    # Alg 4, Line 8: Tìm m_tilde sao cho thỏa điều kiện thứ tự
    # |W_(j,m)| >= w_{j,m} >= |W_(j,m+1)|
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    # Xác định chỉ số m thỏa mãn điều kiện
    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    # Chọn giá trị tối ưu tương ứng với m_tilde
    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    # Alg 4, Line 9: Cập nhật theta_j (beta_star)
    # theta_j <- (1/M) * w_{j, m_tilde} * (theta_j / ||theta_j||)
    # Code: v * x_star = v * (w_star / (M * norm_v)) = (1/M) * w_star * (v / norm_v) (đúng công thức)
    beta_star = x_star * v
    
    # Alg 4, Line 10: Cập nhật W_j (theta_star)
    # W_j <- sign(W_j) * min(w_{j, m_tilde}, |W_j|)
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star


def inplace_prox(beta, theta, lambda_, lambda_bar, M):
    beta.weight.data, theta.weight.data = prox(
        beta.weight.data, theta.weight.data, lambda_=lambda_, lambda_bar=lambda_bar, M=M
    )


def inplace_group_prox(groups, beta, theta, lambda_, lambda_bar, M):
    """
    groups is an iterable such that group[i] contains the indices of features in group i
    """
    beta_ = beta.weight.data
    theta_ = theta.weight.data
    beta_ans = torch.empty_like(beta_)
    theta_ans = torch.empty_like(theta_)
    for g in groups:
        group_beta = beta_[:, g]
        group_beta_shape = group_beta.shape
        group_theta = theta_[:, g]
        group_theta_shape = group_theta.shape
        group_beta, group_theta = prox(
            group_beta.reshape(-1),
            group_theta.reshape(-1),
            lambda_=lambda_,
            lambda_bar=lambda_bar,
            M=M,
        )
        beta_ans[:, g] = group_beta.reshape(*group_beta_shape)
        theta_ans[:, g] = group_theta.reshape(*group_theta_shape)
    beta.weight.data, theta.weight.data = beta_ans, theta_ans

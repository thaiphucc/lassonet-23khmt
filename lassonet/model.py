import torch.nn.functional as F
import torch
from torch import nn
from .prox import prox

# https://github.com/lasso-net/lassonet/blob/master/lassonet/model.py#L10
# LassoNet: Mạng nơ-ron với kết nối thưa (sparse connections) để lựa chọn đặc trưng.
# Kế thừa từ nn.Module của PyTorch.
class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None):
        assert len(dims) > 2
        if groups is not None:
            n_inputs = dims[0]
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups

        super().__init__()
        
        # Các lớp ẩn (nonlinear part) của mạng
        # Đây là phần "mạng dư" (residual) G(x) trong bài báo
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        # Lớp skip connection (linear part) W^T x
        # Đây là thành phần chính để lựa chọn đặc trưng.
        # Trọng số của lớp này sẽ bị ràng buộc bởi norm của lớp ẩn đầu tiên.
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        # Lan truyền thẳng (feed-forward)
        
        current_layer = inp
        # Tính kết quả từ skip connection: W^T x
        result = self.skip(inp)
        
        # Tính kết quả qua các lớp ẩn: G(x)
        for theta in self.layers:
            current_layer = theta(current_layer)
            # Áp dụng activation và dropout cho các lớp trừ lớp cuối cùng
            if theta is not self.layers[-1]:
                current_layer = F.relu(current_layer)
                
        # Kết quả cuối cùng là tổng của skip connection và mạng ẩn: W^T x + G(x)
        return result + current_layer

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=1, dim=0).sum()

    def lambda_start(
        self,
        M=1,
        lambda_bar=0,
        factor=2,
    ):
        """
        Ước lượng giá trị lambda bắt đầu (lambda_max) khi toàn bộ các đặc trưng bị loại bỏ (sparse).
        Hàm này tìm lambda sao cho tất cả trọng số của lớp skip về 0.
        """

        def is_sparse(lambda_):
            with torch.no_grad():
                v = self.skip.weight.data
                u = self.layers[0].weight.data

                for _ in range(10000):
                    # Giải bài toán tối ưu hóa - proximal operator
                    # Cập nhật beta (trọng số skip) và theta (trọng số lớp ẩn đầu)
                    # tuân theo ràng buộc phân cấp (hierarchy constraint): |W_j| <= M * ||V_j||
                    new_v, new_u = prox(
                        v,
                        u,
                        lambda_=lambda_,
                        lambda_bar=lambda_bar,
                        M=M,
                    )
                    # Kiểm tra hội tụ
                    if torch.abs(v - new_v).max() < 1e-5:
                        break
                    v = new_v
                return (torch.norm(v, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

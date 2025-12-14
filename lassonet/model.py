import torch.nn.functional as F
from torch import nn

# https://github.com/lasso-net/lassonet/blob/master/lassonet/model.py#L10
class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None, dropout=None):
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

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)
        return result + current_layer

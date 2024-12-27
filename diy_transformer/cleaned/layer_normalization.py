class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha (multiplicative) is a learnable parameter
        self.bias = nn.Parameter(
            torch.zeros(features)
        )  # bias (additive) is a learnable parameter

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
import torch
import torch.nn as nn

from tabular.utils.utils import cprint


class TabularEncoderBackbone(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads_factor: int = 64,
                 ffn_d_hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        dim_feedforward = d_model * ffn_d_hidden_multiplier
        num_heads = d_model // num_heads_factor
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        cprint(f"❗ Our encoder has {num_layers} layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

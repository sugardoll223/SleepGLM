from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from s4torch import S4Model


def concat_on_seq_len(output: list[torch.Tensor], channel_first: bool = True) -> torch.Tensor:
    """
    Concatenate on sequence axis.
    - [B, C, L] -> dim=-1
    - [B, L, C] -> dim=1
    """
    if len(output) == 0:
        raise ValueError("`output` is empty.")
    return torch.cat(output, dim=(-1 if channel_first else 1))


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    """Conv -> S4Model -> Pool"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        n_layers: int = 2,
        l_max: int = 120,
        pool_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.l_max = max(2, int(l_max))
        self.pool_size = int(pool_size)
        self.conv_block = ConvBlock1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
        )
        self.s4_block = S4Model(
            d_input=hidden_channels,
            d_model=hidden_channels,
            d_output=hidden_channels,
            n_blocks=n_layers,
            n=hidden_channels,
            l_max=self.l_max,
            collapse=False,
        )
        if self.pool_size > 1:
            self.pool = nn.AvgPool1d(kernel_size=self.pool_size, stride=self.pool_size)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        current_len = int(x.shape[-1])
        if current_len < self.l_max:
            pad = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.l_max - current_len,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=-1)
        elif current_len > self.l_max:
            x = x[..., : self.l_max]
        x = x.transpose(1, 2).contiguous()
        x = self.s4_block(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        return x


class ModalityEncoder(nn.Module):
    """init_convs -> EncoderBlock stack -> final_conv -> global pool"""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        hidden_channels: int,
        n_blocks: int = 4,
        pool_sizes: list[int] | None = None,
        kernel_size: int = 3,
        seq_len: int = 3000,
        dropout: float = 0.1,
        **_: dict,
    ) -> None:
        super().__init__()

        self.init_convs = nn.Sequential(
            ConvBlock1d(in_channels, hidden_channels, kernel_size, dropout=dropout),
            ConvBlock1d(hidden_channels, hidden_channels, kernel_size, dropout=dropout),
            nn.AdaptiveAvgPool1d(max(2, seq_len // 5)),
            ConvBlock1d(hidden_channels, hidden_channels, kernel_size, dropout=dropout),
            nn.AdaptiveAvgPool1d(max(2, seq_len // 25)),
        )

        seq_len = max(2, seq_len // 25)
        if pool_sizes is None:
            pool_sizes = [2] * n_blocks
        if len(pool_sizes) != n_blocks:
            raise ValueError(
                f"`pool_sizes` length must equal n_blocks. got len(pool_sizes)={len(pool_sizes)}, n_blocks={n_blocks}"
            )

        cum_pool_sizes = [1] + list(np.cumprod(pool_sizes, dtype=int))

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    n_layers=2,
                    l_max=max(2, seq_len // max(1, cum_pool_sizes[i])),
                    pool_size=pool_sizes[i],
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )

        self.final_conv = ConvBlock1d(hidden_channels, d_model, kernel_size, dropout=dropout)
        self.final_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_convs(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.final_pool(x)
        x = x.squeeze(-1)
        return x
    
if __name__ == "__main__":
    # Test the modules with dummy data
    # Test EEG modality encoder
    batch_size = 4
    in_channels = 6
    seq_len = 3000
    d_model = 256
    hidden_channels = 128

    dummy_input = torch.randn(batch_size, in_channels, seq_len)
    encoder = ModalityEncoder(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        n_blocks=4,
        pool_sizes=[2, 2, 2, 3],
        kernel_size=3,
        seq_len=seq_len,
        dropout=0.1,
    )
    output = encoder(dummy_input)
    print("Output shape:", output.shape)
    
    # Test other modality encoder (e.g., EMG with 1 channels 10Hz)
    in_channels_emg = 1
    seq_len_emg = 300 
    dummy_input_emg = torch.randn(batch_size, in_channels_emg, seq_len_emg)
    encoder_emg = ModalityEncoder(
        in_channels=in_channels_emg,
        d_model=d_model,
        hidden_channels=hidden_channels,
        n_blocks=2,
        pool_sizes=[2, 2],
        kernel_size=3,
        seq_len=seq_len_emg,
        dropout=0.1,
    )
    output_emg = encoder_emg(dummy_input_emg)
    print("EMG Output shape:", output_emg.shape)

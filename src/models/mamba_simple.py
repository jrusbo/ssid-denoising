import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union


class MambaSimple(nn.Module):
    """A pure PyTorch implementation of the Mamba block.

    This is slower than the optimized CUDA version but produces
    mathematically identical results and works on CPU/Windows without mamba-ssm.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        **kwargs,
    ) -> None:
        """Initializes MambaSimple.

        Args:
            d_model: Dimension of the input model.
            d_state: State dimension.
            d_conv: Convolution kernel size.
            expand: Expansion factor.
            dt_rank: Rank of the delta projection. If "auto", calculates based on d_model.
            dt_min: Minimum value for delta.
            dt_max: Maximum value for delta.
            dt_init: Initialization method for delta projection ("random" or "constant").
            dt_scale: Scaling factor for delta initialization.
            dt_init_floor: Floor value for delta initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = (self.d_model + 15) // 16
        else:
            self.dt_rank = int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
            + torch.log(torch.tensor(dt_min))
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D).
        """
        (b, seq_len, d) = x.shape

        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        # Conv1d path
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # SSM path
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)  # (B, L, D_inner)
        dt = F.softplus(dt)

        # Discretization
        A = -torch.exp(self.A_log.float())  # (D_inner, d_state)

        # Parallel scan (Simplified version)
        y = self.selective_scan(x, dt, A, B, C, self.D)

        # Gate with Z
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Naive sequential implementation of selective scan.

        Used for fallback when optimized kernels are unavailable.

        Args:
            u: Input tensor of shape (B, L, D_inner).
            dt: Delta tensor of shape (B, L, D_inner).
            A: State transition matrix of shape (D_inner, d_state).
            B: Input transition matrix of shape (B, L, d_state).
            C: Output transition matrix of shape (B, L, d_state).
            D: Direct pass-through vector of shape (D_inner).

        Returns:
            Output tensor of shape (B, L, D_inner).
        """
        (b, seq_len, d_in) = u.shape
        n = A.shape[1]

        # Discretize A and B
        dt = dt.unsqueeze(-1)  # (B, L, D_inner, 1)
        A = A.view(1, 1, d_in, n)  # (1, 1, D_inner, d_state)
        dA = torch.exp(dt * A)  # (B, L, D_inner, d_state)

        B = B.unsqueeze(2)  # (B, L, 1, d_state)
        dB = dt * B  # (B, L, D_inner, d_state)

        # Scan
        h = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(seq_len):
            h = dA[:, i] * h + dB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D_inner)
        y = y + u * D
        return y

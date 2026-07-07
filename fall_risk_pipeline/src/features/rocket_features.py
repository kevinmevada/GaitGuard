"""
ROCKET / MINIROCKET random convolution features (Dempster 2019/2021).

Each kernel yields:
  - max convolution value
  - proportion of positive values (PPV) above bias threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Kernel:
    length: int
    weights: np.ndarray
    bias: float
    dilation: int
    padding: int
    channel: int


def _valid_length(signal_len: int, kernel_len: int, dilation: int, padding: int) -> int:
    effective = (kernel_len - 1) * dilation + 1
    return signal_len + 2 * padding - effective + 1


def _conv1d_channel(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float,
    dilation: int,
    padding: int,
) -> np.ndarray:
    """1D convolution for a single channel (valid + manual padding)."""
    if padding > 0:
        x = np.pad(x, (padding, padding), mode="edge")
    # x is already padded above, so use padding=0 here to get the length of
    # the padded signal directly (single source of truth for the formula).
    L = len(x)
    k = len(weights)
    n_out = _valid_length(L, k, dilation, padding=0)
    if n_out <= 0:
        return np.array([0.0], dtype=float)
    out = np.empty(n_out, dtype=float)
    for i in range(n_out):
        acc = bias
        for j, w in enumerate(weights):
            acc += w * x[i + j * dilation]
        out[i] = acc
    return out


class RocketTransform:
    """ROCKET with random real-valued kernels (default 10,000)."""

    def __init__(
        self,
        n_kernels: int = 10_000,
        *,
        seed: int = 42,
        kernel_lengths: tuple[int, ...] = (7, 9, 11),
    ):
        self.n_kernels = int(n_kernels)
        self.seed = int(seed)
        self.kernel_lengths = kernel_lengths
        self.kernels: list[_Kernel] = []
        self.n_channels: int | None = None

    def fit(self, X: np.ndarray) -> RocketTransform:
        """Generate random kernels (unsupervised). X shape: (N, C, T)."""
        if X.ndim != 3:
            raise ValueError("X must be (N, C, T)")
        self.n_channels = int(X.shape[1])
        rng = np.random.default_rng(self.seed)
        self.kernels = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(self.kernel_lengths))
            weights = rng.normal(size=length).astype(np.float64)
            weights -= weights.mean()
            bias = float(rng.uniform(-1.0, 1.0))
            dilation = int(rng.choice([1, 2, 4, 8]))
            padding = int(rng.choice([0, length // 2]))
            channel = int(rng.integers(0, self.n_channels))
            self.kernels.append(_Kernel(length, weights, bias, dilation, padding, channel))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2 * n_kernels): [max, ppv] per kernel."""
        if not self.kernels:
            raise RuntimeError("Call fit() before transform()")
        n = X.shape[0]
        out = np.empty((n, 2 * len(self.kernels)), dtype=np.float32)
        for i in range(n):
            out[i] = self._transform_one(X[i])
        return out

    def _transform_one(self, x: np.ndarray) -> np.ndarray:
        feats: list[float] = []
        for k in self.kernels:
            conv = _conv1d_channel(x[k.channel], k.weights, k.bias, k.dilation, k.padding)
            feats.append(float(np.max(conv)))
            feats.append(float(np.mean(conv > 0.0)))
        return np.asarray(feats, dtype=np.float32)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            n_kernels=self.n_kernels,
            seed=self.seed,
            n_channels=self.n_channels or -1,
            lengths=np.array([k.length for k in self.kernels], dtype=np.int16),
            weights=np.array([k.weights.astype(np.float32) for k in self.kernels], dtype=object),
            biases=np.array([k.bias for k in self.kernels], dtype=np.float32),
            dilations=np.array([k.dilation for k in self.kernels], dtype=np.int16),
            paddings=np.array([k.padding for k in self.kernels], dtype=np.int16),
            channels=np.array([k.channel for k in self.kernels], dtype=np.int16),
            variant=np.array("rocket"),
        )

    @classmethod
    def load(cls, path) -> RocketTransform:
        data = np.load(path, allow_pickle=True)
        obj = cls(int(data["n_kernels"]), seed=int(data["seed"]))
        obj.n_channels = int(data["n_channels"])
        obj.kernels = []
        for i in range(int(data["n_kernels"])):
            obj.kernels.append(
                _Kernel(
                    int(data["lengths"][i]),
                    np.asarray(data["weights"][i], dtype=np.float64),
                    float(data["biases"][i]),
                    int(data["dilations"][i]),
                    int(data["paddings"][i]),
                    int(data["channels"][i]),
                )
            )
        return obj


class MiniRocketTransform(RocketTransform):
    """MINIROCKET: deterministic {-1, 2} weight kernels (Dempster 2021)."""

    def fit(self, X: np.ndarray) -> MiniRocketTransform:
        if X.ndim != 3:
            raise ValueError("X must be (N, C, T)")
        self.n_channels = int(X.shape[1])
        rng = np.random.default_rng(self.seed)
        self.kernels = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(self.kernel_lengths))
            weights = rng.choice(np.array([-1.0, 2.0]), size=length).astype(np.float64)
            bias = 0.0
            dilation = int(rng.choice([1, 2, 4, 8]))
            padding = int(rng.choice([0, length // 2]))
            channel = int(rng.integers(0, self.n_channels))
            self.kernels.append(_Kernel(length, weights, bias, dilation, padding, channel))
        return self

    def save(self, path) -> None:
        super().save(path)

    @classmethod
    def load(cls, path) -> MiniRocketTransform:
        rocket = RocketTransform.load(path)
        mini = cls(rocket.n_kernels, seed=rocket.seed)
        mini.n_channels = rocket.n_channels
        mini.kernels = rocket.kernels
        return mini

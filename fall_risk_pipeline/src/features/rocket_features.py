"""
ROCKET / MINIROCKET random convolution features (Dempster 2019/2021).

Each kernel yields:
  - max convolution value
  - proportion of positive values (PPV) above bias threshold

Implementation notes (perf):

1. Kernels are grouped by their (channel, dilation, padding, length)
   signature. Within each group, the convolution for *every window in
   the batch* and *every kernel in the group* is computed as a single
   batched matrix multiply (via sliding_window_view + einsum), instead
   of a naive nested-Python-loop-per-kernel-per-window approach.

2. NEW: the per-group weight matrix `W` (n_group, length) and bias
   vector `B` (n_group,) are now built ONCE, right after fit()/load(),
   and cached on the object (`self._group_matrices`). Previously these
   were rebuilt from `self.kernels` via `np.stack([...])` inside
   `transform()` on *every single call* — harmless for one call, but
   catastrophic when `transform()` is invoked once per trial across
   thousands of trials, since the same ~150-300 group matrices were
   being reconstructed from Python lists thousands of times over.

3. NEW: `transform()` now processes windows in chunks (`batch_size`)
   so that callers can pass in a very large concatenated array (e.g.
   *all* windows from *all* trials at once) without the intermediate
   `(N, n_out, n_group)` einsum output blowing up memory. This is what
   makes it safe to stop calling transform() per-trial and instead
   call it once (or a few times) for the whole dataset — see
   phase3_deep.py's batched extraction path.

Both changes are purely computational bookkeeping — they do not change
kernel generation, the convolution formula, or aggregation, so outputs
are numerically identical (up to negligible floating-point
associativity differences from batching) to the previous
implementation. See tests/test_phase3_deep.py and
tests/test_rocket_features_vectorized.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class _Kernel:
    length: int
    weights: np.ndarray
    bias: float
    dilation: int
    padding: int
    channel: int


@dataclass
class _GroupMatrices:
    """Precomputed, ready-to-use arrays for one (channel, dilation,
    padding, length) group. Built once; reused across every
    transform() call for the lifetime of the fitted transform."""

    idxs: np.ndarray          # (n_group,) kernel indices in original order
    W: np.ndarray             # (n_group, length) weight matrix
    B: np.ndarray             # (n_group,) bias vector


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
    """1D convolution for a single channel (valid + manual padding).

    Kept for reference/compatibility (e.g. ad-hoc debugging of a single
    kernel) — the hot path in transform() below no longer calls this
    function; see RocketTransform.transform().
    """
    if padding > 0:
        x = np.pad(x, (padding, padding), mode="edge")
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


def _group_key(k: "_Kernel") -> tuple[int, int, int, int]:
    return (k.channel, k.dilation, k.padding, k.length)


class RocketTransform:
    """ROCKET with random real-valued kernels (default 10,000)."""

    #: Default number of windows processed per internal chunk inside
    #: transform(). Chosen to keep the largest intermediate tensor
    #: (N_chunk, n_out, n_group) comfortably within a few hundred MB
    #: for typical n_out/n_group sizes. Override via transform(...,
    #: batch_size=...) if you know your memory budget is tighter/looser.
    DEFAULT_TRANSFORM_BATCH_SIZE = 8192

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
        self._groups_cache: dict[tuple[int, int, int, int], list[int]] | None = None
        self._group_matrices: dict[tuple[int, int, int, int], _GroupMatrices] | None = None

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
        self._invalidate_caches()
        self._build_group_matrices()
        return self

    def _invalidate_caches(self) -> None:
        self._groups_cache = None
        self._group_matrices = None

    def _build_group_matrices(self) -> None:
        """Group kernels by (channel, dilation, padding, length) and
        precompute the stacked weight/bias matrices for each group.

        This runs exactly once per fit()/load() call, regardless of how
        many times transform() is subsequently invoked — that's the
        whole point. Previously this stacking happened inside
        transform() itself, once per group per call.
        """
        groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for idx, k in enumerate(self.kernels):
            groups[_group_key(k)].append(idx)

        group_matrices: dict[tuple[int, int, int, int], _GroupMatrices] = {}
        for key, idxs in groups.items():
            idxs_arr = np.asarray(idxs, dtype=np.int64)
            W = np.stack([self.kernels[i].weights for i in idxs])  # (n_group, length)
            B = np.array([self.kernels[i].bias for i in idxs])     # (n_group,)
            group_matrices[key] = _GroupMatrices(idxs=idxs_arr, W=W, B=B)

        self._groups_cache = dict(groups)
        self._group_matrices = group_matrices

    def _ensure_group_matrices(self) -> dict[tuple[int, int, int, int], _GroupMatrices]:
        if self._group_matrices is None:
            self._build_group_matrices()
        assert self._group_matrices is not None
        return self._group_matrices

    def transform(self, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        """Return (N, 2 * n_kernels): [max, ppv] per kernel.

        `X` can be the windows for a single trial OR the concatenated
        windows for many trials (even the whole dataset) — pass the
        largest batch you can fit in memory; larger batches amortize
        Python/dispatch overhead better. Internally chunked by
        `batch_size` (default DEFAULT_TRANSFORM_BATCH_SIZE windows) so
        arbitrarily large X is safe to pass in one call.

        Numerically equivalent to the naive per-window, per-kernel loop
        this replaces, and equivalent to calling transform() separately
        on sub-batches and concatenating the results (each window's
        output depends only on that window, never on other windows in
        the batch).
        """
        if not self.kernels:
            raise RuntimeError("Call fit() before transform()")
        group_matrices = self._ensure_group_matrices()

        n = X.shape[0]
        bs = batch_size or self.DEFAULT_TRANSFORM_BATCH_SIZE
        if n <= bs:
            return self._transform_chunk(X, group_matrices)

        chunks = []
        for start in range(0, n, bs):
            chunks.append(self._transform_chunk(X[start:start + bs], group_matrices))
        return np.concatenate(chunks, axis=0)

    def _transform_chunk(
        self,
        X: np.ndarray,
        group_matrices: dict[tuple[int, int, int, int], _GroupMatrices],
    ) -> np.ndarray:
        n = X.shape[0]
        out = np.empty((n, 2 * len(self.kernels)), dtype=np.float32)

        for (channel, dilation, padding, length), gm in group_matrices.items():
            xc = X[:, channel, :]  # (N, T) — this group's channel, all windows at once
            if padding > 0:
                xc = np.pad(xc, ((0, 0), (padding, padding)), mode="edge")
            T = xc.shape[1]
            n_out = _valid_length(T, length, dilation, padding=0)

            if n_out <= 0:
                out[:, 2 * gm.idxs] = 0.0
                out[:, 2 * gm.idxs + 1] = 0.0
                continue

            effective = (length - 1) * dilation + 1
            windows_view = sliding_window_view(xc, effective, axis=1)[:, :n_out, :]
            if dilation > 1:
                positions = np.arange(0, effective, dilation)
                windows_view = windows_view[:, :, positions]  # (N, n_out, length)

            # (N, n_out, length) @ (length, n_group) -> (N, n_out, n_group)
            conv_all = np.einsum("wol,gl->wog", windows_view, gm.W, optimize=True) + gm.B

            maxes = conv_all.max(axis=1)             # (N, n_group)
            ppvs = (conv_all > 0.0).mean(axis=1)      # (N, n_group)

            out[:, 2 * gm.idxs] = maxes
            out[:, 2 * gm.idxs + 1] = ppvs

        return out

    def _transform_one(self, x: np.ndarray) -> np.ndarray:
        """Single-window convenience wrapper around transform() — kept for
        any external/debugging callers that used the old per-window API."""
        return self.transform(x[np.newaxis, ...])[0]

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
        obj._invalidate_caches()
        obj._build_group_matrices()
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
        self._invalidate_caches()
        self._build_group_matrices()
        return self

    def save(self, path) -> None:
        super().save(path)

    @classmethod
    def load(cls, path) -> MiniRocketTransform:
        rocket = RocketTransform.load(path)
        mini = cls(rocket.n_kernels, seed=rocket.seed)
        mini.n_channels = rocket.n_channels
        mini.kernels = rocket.kernels
        mini._invalidate_caches()
        mini._build_group_matrices()
        return mini
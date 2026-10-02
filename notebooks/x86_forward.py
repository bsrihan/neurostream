"""Forward spike-band filter compiled for x86-64 AVX-512.

Journal of Neuroscience Methods papers give the update, not only the
library call. This is that update for the forward half of the
zero-phase spike-band filter (direct form II transposed, one
second-order section at a time)::

    y[n] = b0 * x[n] + z1
    z1   = b1 * x[n] - a1 * y[n] + z2
    z2   = b2 * x[n] - a2 * y[n]

Time is a recurrence, so a wider SIMD register does not shorten one
channel. The channels are independent, and the compiled loop keeps
them contiguous so an AVX-512 register holds eight channels at once.
Coefficients stay float64. The result is stored the same way SciPy's
``sosfilt`` is stored (float64 arithmetic, then the caller's dtype).

A parallel OpenMP loop is not used. On this processor the reverse
filter is an OpenBLAS multiply, and the two libraries share one
thread pool. Running them in the same process made the multiply
slower. The compiled loop is single-threaded and uses the wide
registers instead.

Machines without AVX-512 keep ``scipy.signal.sosfilt``.
"""

from __future__ import annotations

import os
import platform
import sys
import warnings

import numpy as np

# Skylake-AVX512 is the LLVM target that emits zmm code. It has to be
# chosen before Numba is imported. ``-prefer-256-bit`` turns off LLVM's
# habit of keeping AVX-512 CPUs on 256-bit registers.
_AVX512_FLAGS = frozenset(("avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl"))
_KERNEL = None
_KERNEL_READY = False


def cpu_supports_avx512():
    """True when this process can run a Skylake-class AVX-512 loop."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    return _AVX512_FLAGS <= _cpu_flags()


def _cpu_flags():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        return frozenset()
    return frozenset()


def _prepare_numba_target():
    """Point Numba at AVX-512 if it has not already chosen a target."""
    if not cpu_supports_avx512():
        return False
    if "numba" in sys.modules:
        name = getattr(sys.modules["numba"].config, "CPU_NAME", None)
        return name in (None, "skylake-avx512")
    os.environ.setdefault("NUMBA_CPU_NAME", "skylake-avx512")
    os.environ.setdefault("NUMBA_CPU_FEATURES", "-prefer-256-bit")
    return True


def _compile_kernel():
    """Return the jitted recurrence, or None when it is not AVX-512."""
    if not _prepare_numba_target():
        return None
    from numba import njit

    @njit(nogil=True, cache=True)
    def sos_time_major(norm, xt, z1, z2):
        n_samples, n_channels = xt.shape
        n_sections = norm.shape[0]
        for section in range(n_sections):
            b0 = norm[section, 0]
            b1 = norm[section, 1]
            b2 = norm[section, 2]
            a1 = norm[section, 3]
            a2 = norm[section, 4]
            for sample in range(n_samples):
                for channel in range(n_channels):
                    xn = xt[sample, channel]
                    yn = b0 * xn + z1[section, channel]
                    z1[section, channel] = b1 * xn - a1 * yn + z2[section, channel]
                    z2[section, channel] = b2 * xn - a2 * yn
                    xt[sample, channel] = yn

    # Compile on a wide channel axis so the vectorizer emits zmm, then
    # refuse the kernel if the assembly stayed on narrower registers.
    norm = np.zeros((2, 5), dtype=np.float64)
    norm[:, 0] = 1.0
    xt = np.zeros((8, 64), dtype=np.float64)
    z1 = np.zeros((2, 64), dtype=np.float64)
    z2 = np.zeros((2, 64), dtype=np.float64)
    # A cache hit cannot dump assembly (Numba returns an empty module and
    # warns). The cache key includes the skylake-avx512 target, so a hit
    # on this CPU is the 512-bit kernel. A fresh compile must show zmm.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sos_time_major(norm, xt, z1, z2)
        assembly = "\n".join(sos_time_major.inspect_asm().values())
    if "zmm" in assembly:
        return sos_time_major
    cached = any("Inspection disabled" in str(item.message) for item in caught)
    from numba.core import config
    if cached and config.CPU_NAME == "skylake-avx512" and cpu_supports_avx512():
        return sos_time_major
    return None


def _kernel():
    global _KERNEL, _KERNEL_READY
    if not _KERNEL_READY:
        _KERNEL_READY = True
        try:
            _KERNEL = _compile_kernel()
        except Exception:
            _KERNEL = None
    return _KERNEL


class ForwardSOS:
    """Streaming forward filter. One instance per processor.

    ``sos`` is the SciPy second-order section matrix, shape
    ``(n_sections, 6)``. ``zi`` is the SciPy initial state, shape
    ``(n_sections, n_channels, 2)``, copied in so the caller's array
    stays put. State after that lives in this object.
    """

    def __init__(self, sos, zi, n_samples, kernel):
        sos = np.asarray(sos, dtype=np.float64)
        zi = np.asarray(zi, dtype=np.float64)
        if zi.ndim != 3 or zi.shape[0] != sos.shape[0] or zi.shape[2] != 2:
            raise ValueError(
                "zi must have shape (n_sections, n_channels, 2), "
                f"got {zi.shape}")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        norm = np.empty((sos.shape[0], 5), dtype=np.float64)
        for section in range(sos.shape[0]):
            inverse_a0 = 1.0 / sos[section, 3]
            norm[section, 0] = sos[section, 0] * inverse_a0
            norm[section, 1] = sos[section, 1] * inverse_a0
            norm[section, 2] = sos[section, 2] * inverse_a0
            norm[section, 3] = sos[section, 4] * inverse_a0
            norm[section, 4] = sos[section, 5] * inverse_a0
        n_channels = zi.shape[1]
        self._kernel = kernel
        self._norm = np.ascontiguousarray(norm)
        self._z1 = np.ascontiguousarray(zi[:, :, 0])
        self._z2 = np.ascontiguousarray(zi[:, :, 1])
        self._xt = np.empty((n_samples, n_channels), dtype=np.float64)
        self.n_samples = int(n_samples)
        self.n_channels = int(n_channels)

    def apply(self, data, dest):
        """Filter ``data`` of shape ``(n_channels, n_samples)`` into ``dest``.

        ``dest`` may be float32. The arithmetic is float64, and the
        store casts, matching ``sosfilt`` followed by an assignment
        into a float32 buffer.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.shape != (self.n_channels, self.n_samples):
            raise ValueError(
                f"expected shape {(self.n_channels, self.n_samples)}, "
                f"got {data.shape}")
        # Channel axis contiguous: sample ``n`` is one SIMD-friendly row.
        np.copyto(self._xt, np.transpose(data))
        self._kernel(self._norm, self._xt, self._z1, self._z2)
        dest[:, :] = np.transpose(self._xt)


def try_create_forward(sos, zi, n_samples):
    """Build an AVX-512 forward filter, or return None to keep SciPy.

    None means the CPU lacks AVX-512, Numba was already imported for a
    different target, or the compiled loop did not use 512-bit registers.
    """
    kernel = _kernel()
    if kernel is None:
        return None
    return ForwardSOS(sos, zi, n_samples, kernel)

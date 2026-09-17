"""Benchmark-only kernels used for controlled experimental validation.

Nothing in this package is a production MPC implementation.
"""

from .minimal_shamir_kernel import MinimalShamirKernel, KernelCounters

__all__ = ["MinimalShamirKernel", "KernelCounters"]

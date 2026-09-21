import numpy as np

from pdr.matvec import PlaintextBackend


def test_plaintext_backend_matches_numpy_matmul():
    rng = np.random.default_rng(1)
    M = rng.integers(-127, 128, size=(20, 64), dtype=np.int64)
    x = rng.integers(-127, 128, size=64, dtype=np.int64)
    backend = PlaintextBackend()
    got = backend.matvec(M, x)
    want = M @ x
    assert np.array_equal(got, want)


def test_plaintext_backend_returns_int64():
    backend = PlaintextBackend()
    M = np.array([[1, 2], [3, 4]], dtype=np.int8)
    x = np.array([5, 6], dtype=np.int8)
    out = backend.matvec(M, x)
    assert out.dtype == np.int64
    assert np.array_equal(out, np.array([17, 39]))

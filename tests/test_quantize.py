import numpy as np

from pdr.quantize import quantize, symmetric_scale


def test_quantize_range_int8():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 16)).astype(np.float32)
    Xi, s = quantize(X, bits=8)
    assert Xi.min() >= -127 and Xi.max() <= 127
    assert s > 0
    recon = Xi * s
    assert np.max(np.abs(recon - X)) < 2 * s


def test_quantize_zero_array():
    X = np.zeros((4, 4), dtype=np.float32)
    Xi, s = quantize(X, bits=8)
    assert np.all(Xi == 0)
    assert s == 1.0


def test_symmetric_scale_picks_amax():
    X = np.array([[-3.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    s = symmetric_scale(X, bits=8)
    assert np.isclose(s, 3.0 / 127.0)


def test_clipped_scale_saturates_outliers():
    from pdr.quantize import clipped_scale, quantize

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 64)).astype(np.float32) * 0.03
    X[0, 0] = 5.0  # one outlier sets the max-abs step
    s_max = quantize(X, bits=4)[1]
    s_clip = clipped_scale(X, bits=4, percentile=99)
    assert s_clip < s_max / 10
    Xi, s = quantize(X, bits=4, clip_percentile=99)
    assert s == s_clip
    assert Xi.max() == 7 and Xi.min() >= -7  # saturated at qmax, still inside [-qmax, qmax]
    assert Xi[0, 0] == 7
    assert 0.005 < np.mean(np.abs(Xi) == 7) < 0.05  # about 1% of entries saturate


def test_clip_100_is_maxabs():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 8)).astype(np.float32)
    a, sa = quantize(X, bits=6)
    b, sb = quantize(X, bits=6, clip_percentile=100)
    assert sa == sb and np.array_equal(a, b)

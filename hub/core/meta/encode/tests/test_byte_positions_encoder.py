import numpy as np
import pytest
from hub.core.meta.encode.byte_positions import BytePositionsEncoder


def test_trivial():
    enc = BytePositionsEncoder()

    np.testing.assert_array_equal(enc._encoded, np.array([], dtype=np.uint64))

    assert enc.num_samples == 0

    enc.register_samples(8, 100)
    enc.register_samples(8, 100)

    assert enc.num_samples == 200
    assert len(enc._encoded) == 1
    assert enc.num_bytes_encoded_under_row(-1) == 1600

    enc.register_samples(1, 1000)

    assert enc.num_samples == 1200
    assert len(enc._encoded) == 2
    assert enc.num_bytes_encoded_under_row(-1) == 2600

    assert enc[0] == (0, 8)
    assert enc[1] == (8, 16)
    assert enc[199] == (1592, 1600)
    assert enc[200] == (1600, 1601)
    assert enc[201] == (1601, 1602)
    assert enc[1199] == (2599, 2600)

    enc.register_samples(16, 32)

    assert enc.num_samples == 1232
    assert len(enc._encoded) == 3
    assert enc.num_bytes_encoded_under_row(-1) == 3112

    assert enc[1200] == (2600, 2616)

    with pytest.raises(IndexError):
        enc[1232]


def test_non_uniform():
    enc = BytePositionsEncoder()

    assert enc.num_samples == 0

    enc.register_samples(4960, 1)
    enc.register_samples(4961, 1)
    enc.register_samples(41, 1)

    assert enc.num_samples == 3
    assert len(enc._encoded) == 3

    assert enc[0] == (0, 4960)
    assert enc[1] == (4960, 4960 + 4961)
    assert enc[2] == (4960 + 4961, 4960 + 4961 + 41)


def test_failures():
    enc = BytePositionsEncoder()

    with pytest.raises(ValueError):
        # num_samples cannot be 0
        enc.register_samples(8, 0)

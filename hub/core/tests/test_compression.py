from hub.tests.common import get_actual_compression_from_buffer
import numpy as np
import pytest
from hub.core.compression import (
    compress_array,
    decompress_array,
    compress_multiple,
    decompress_multiple,
)


parametrize_compressions = pytest.mark.parametrize(
    "compression", ["jpeg", "png"]
)  # TODO: extend to be all pillow types we want to focus on

parametrize_image_shapes = pytest.mark.parametrize(
    "shape",
    [(100, 100, 3), (28, 28, 1), (32, 32)],  # JPEG does not support RGBA
)


@parametrize_compressions
@parametrize_image_shapes
def test_array(compression, shape):
    # TODO: check dtypes and no information loss
    array = np.zeros(shape, dtype="uint8")  # TODO: handle non-uint8
    compressed_buffer = compress_array(array, compression)
    assert get_actual_compression_from_buffer(compressed_buffer) == compression
    decompressed_array = decompress_array(compressed_buffer, shape=shape)
    np.testing.assert_array_equal(array, decompressed_array)


@parametrize_compressions
def test_multi_array(compression):
    shapes = [(100, 100, 3), (100, 50, 3), (50, 100, 3), (50, 50, 3)]
    arrays = [np.ones(shape, dtype="uint8") for shape in shapes]
    compressed_buffer = compress_multiple(arrays, compression)
    decompressed_arrays = decompress_multiple(compressed_buffer, shapes)
    for arr1, arr2 in zip(arrays, decompressed_arrays):
        if compression == "png":
            np.testing.assert_array_equal(arr1, arr2)
        else:
            assert arr1.shape == arr2.shape

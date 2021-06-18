from hub.constants import SUPPORTED_COMPRESSIONS
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
)
from typing import Union
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO


def compress_array(array: np.ndarray, compression: str) -> bytes:
    """Compress some numpy array using `compression`. All meta information will be contained in the returned buffer.

    Note:
        `decompress_array` may be used to decompress from the returned bytes back into the `array`.

    Args:
        array (np.ndarray): Array to be compressed.
        compression (str): `array` will be compressed with this compression into bytes. Right now only arrays compatible with `PIL` will be compressed.

    Raises:
        UnsupportedCompressionError: If `compression` is unsupported. See `SUPPORTED_COMPRESSIONS`.

    Returns:
        bytes: Compressed `array` represented as bytes.
    """

    if compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(compression)

    try:
        img = Image.fromarray(array)
        out = BytesIO()
        img.save(out, compression)
        out.seek(0)
        return out.read()
    except (TypeError, OSError) as e:
        raise SampleCompressionError(array.shape, compression, str(e))


def decompress_array(buffer: Union[bytes, memoryview]) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Note:
        `compress_array` may be used to get the `buffer` input.

    Args:
        buffer (bytes, memoryview): Buffer to be decompressed. It is assumed all meta information required to
            decompress is contained within `buffer`.

    Raises:
        SampleDecompressionError: Right now only buffers compatible with `PIL` will be decompressed.

    Returns:
        np.ndarray: Array from the decompressed buffer.
    """

    try:
        img = Image.open(BytesIO(buffer))
        return np.array(img)
    except UnidentifiedImageError:
        raise SampleDecompressionError()

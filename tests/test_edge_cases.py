"""
Edge-case regression tests against locally committed DICOM fixtures.
"""

from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread

EDGE_CASES_DIR = Path(__file__).parent / "edge-cases"

LOSSLESS_YCBCR_PATH = EDGE_CASES_DIR / "lossless-jpeg-encoded-as-ycbcr.dcm"


def test_lossless_jpeg_with_ycbcr_sof_is_converted_to_rgb():
    """
    GitHub issue #1: a JPEG Lossless (Process 14) file whose SOF marker
    declares YCbCr must be converted to RGB by this plugin — the underlying
    ``jpeg-decoder`` crate does no colour conversion for lossless JPEG, so
    without this we would silently return YCbCr samples labelled as RGB.
    """

    ds = dcmread(LOSSLESS_YCBCR_PATH)
    arr = ds.pixel_array

    assert arr.shape == (434, 636, 3)
    assert arr.dtype == np.uint8

    # The image is an echocardiogram with a coloured Doppler overlay. The
    # following pixels are sampled from that overlay; if YCbCr→RGB conversion
    # is broken (or the bytes are still raw YCbCr) the channel values diverge
    # dramatically from these.
    assert arr[181, 577].tolist() == [0, 255, 0]
    assert arr[18, 342].tolist() == [240, 255, 0]

    means = arr.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(means, [22.498, 23.634, 22.301], atol=0.01)

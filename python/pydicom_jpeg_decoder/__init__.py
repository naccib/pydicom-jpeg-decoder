import logging

from pydicom import uid
from pydicom.pixels.decoders.base import (
    DecodeRunner,
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
)

from .pydicom_jpeg_decoder import decode_jpeg  # noqa: F401

_LIBJPEG_SYNTAXES = [
    uid.JPEGBaseline8Bit,
    uid.JPEGExtended12Bit,
    uid.JPEGLossless,
    uid.JPEGLosslessSV1,
]

_DECODERS_TO_INSTALL = [
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
]

DECODER_DEPENDENCIES = {syntax: ("jpeglib", ">=1.0.2") for syntax in _LIBJPEG_SYNTAXES}
"""
The decoder dependencies for each JPEG syntax in the format pydicom expects.
"""

logger = logging.getLogger(__name__)


def is_available(uid: str) -> bool:
    """Return ``True`` if the decoder has its dependencies met, ``False`` otherwise"""
    return uid in _LIBJPEG_SYNTAXES


def decode_frame(src: bytes, runner: DecodeRunner) -> bytearray:
    """Return the decoded image data in `src` as a :class:`bytearray`."""

    logger.info(
        f"pydicom-jpeg-decoder: Decoding {len(src)} bytes for {runner.transfer_syntax}"
    )

    data = decode_jpeg(src)

    logger.info(f"pydicom-jpeg-decoder: decoded {len(data)} bytes")

    return bytearray(data)


def install_plugins():
    """Install the plugins for the JPEG decoders."""

    for decoder in _DECODERS_TO_INSTALL:
        decoder.add_plugins(
            [
                ("pydicom-jpeg-decoder", ("pydicom_jpeg_decoder", "decode_frame")),
            ]
        )

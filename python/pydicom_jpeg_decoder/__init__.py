import logging
from pathlib import Path
from uuid import uuid4

from pydicom import uid
from pydicom.pixels.common import PhotometricInterpretation as PI
from pydicom.pixels.decoders.base import (
    DecodeRunner,
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
)

from .pydicom_jpeg_decoder import decode_jpeg, determine_color_transform  # noqa: F401

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
    """
    Return the decoded image data in `src` as a :class:`bytearray`.

    `jpeg_decoder` will _always_ return a color transform of `RGB` for 3-channel images. We therefore always set the photometric interpretation to `RGB`. Unfortunately this means we'll ignore whatever the requested photometric interpretation is.

    Args:
        src: The bytes of the JPEG image to decode.
        runner: The runner instance to use to decode the image.

    Returns:
        The decoded image data as a :class:`bytearray`.
    """

    if runner.transfer_syntax == uid.JPEGExtended12Bit and runner.bits_stored != 8:
        raise NotImplementedError(
            "pydicom-jpeg-decoder only supports 8-bit precision JPEG Extended transfer syntax"
        )

    logger.info(f"Decoding {len(src)} bytes for {runner.transfer_syntax}")

    decoded = decode_jpeg(src)

    pi = runner.get_option("photometric_interpretation")
    as_rgb = runner.get_option("as_rgb", False)

    # NOTE: We always set the photometric interpretation to `RGB` for 3-channel images.
    if runner.samples_per_pixel == 3:
        logger.warning("3-channel images are always converted to RGB")
        runner.set_option("photometric_interpretation", PI.RGB)

    logger.debug(
        f"DICOM says PI={pi}, samples_per_pixel={runner.samples_per_pixel}, and pydicom wants as_rgb={as_rgb} so convert_to_rgb={as_rgb and 'YBR' in pi}"
    )
    logger.debug(
        f"Decoder says color_transform={decoded.color_transform} and adobe_color_transform={decoded.adobe_color_transform}, so determined_color_transform={decoded.determined_color_transform}"
    )

    data = decoded.pixel_data

    logger.info(f"Decoded {len(data)} bytes")

    return bytearray(data)


def install_plugins(remove_existing: bool = False):
    """
    Install the plugins for the JPEG decoders.

    Args:
        remove_existing: Whether to remove the existing plugins for the JPEG transfer syntaxes supported before installing the new ones.
            Defaults to False.
    """

    for decoder in _DECODERS_TO_INSTALL:
        previously_available_plugins = decoder.available_plugins

        if remove_existing and len(previously_available_plugins) > 0:
            for plugin in previously_available_plugins:
                decoder.remove_plugin(plugin)

                logger.info(f"{decoder.UID}: removed plugin {plugin}")

        decoder.add_plugins(
            [
                ("pydicom-jpeg-decoder", ("pydicom_jpeg_decoder", "decode_frame")),
            ]
        )

import asyncio
from io import BytesIO
from logging import basicConfig
from pathlib import Path

from httpx import AsyncClient
from pydicom.pixels.decoders.base import (
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
)

RELEVANT_DECODERS = [
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
]

CONTROL_IMAGES = [
    "ljdata/ds/JPEGLosslessSV1/532_JPEGLossless_VOI.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEG-LL.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_1s_1f_u_08_08.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_1s_1f_u_16_16.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_3s_2f_u_08_08.dcm",
    "ljdata/ds/JPEGLosslessSV1/MG1_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/RG1_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/RG2_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/SC_rgb_jpeg_gdcm.dcm",
    "ljdata/ds/JPEGLossless/JPEGLossless_1s_1f_u_16_12.dcm",
    "ljdata/ds/JPEGBaseline/JPEGBaseline_1s_1f_u_08_08.dcm",
    "ljdata/ds/JPEGBaseline/SC_jpeg_no_color_transform_2.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cr.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cy+n1.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cy+n2.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cy+np.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cy+s2.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_dcmtk_+eb+cy+s4.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_jpeg_dcmtk.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_jpeg_lossy_gdcm.dcm",
    "ljdata/ds/JPEGBaseline/SC_rgb_small_odd_jpeg.dcm",
    "ljdata/ds/JPEGBaseline/color3d_jpeg_baseline.dcm",
    "ljdata/ds/JPEGExtended/JPEGExtended_3s_1f_u_08_08.dcm",
]

CLAIMS_TO_BE_RGB_BUT_IS_NOT_PATH = Path(
    "tests/edge-cases/claims-to-be-rgb-but-is-not.dcm"
)

CLAIMS_TO_BE_RGB_AND_ACTUALLY_IS_BUT_HOROS_FUCKS_IT_UP_PATH = Path(
    "tests/edge-cases/claims-to-be-rgb-and-actually-is-but-horos-fucks-it-up.dcm"
)

basicConfig(level="INFO")


async def fetch_test_file_from_pylibjpeg_data(path: str) -> bytes:
    """
    Fetch a test file from the [pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) repository.
    """

    async with AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            f"https://github.com/pydicom/pylibjpeg-data/raw/main/{path}"
        )

        response.raise_for_status()

        assert response.content is not None, f"Failed to fetch {path}: no content"
        assert len(response.content) > 0, f"Failed to fetch {path}: empty content"

        return response.content


async def fetch_and_decode_dicom(file_path: str):
    """Fetch and decode a single DICOM file."""
    data = await fetch_test_file_from_pylibjpeg_data(file_path)
    buffer = BytesIO(data)
    from pydicom import dcmread

    ds = dcmread(buffer)
    return ds, file_path


def decode_local_dicom(file_path: Path):
    """Decode a local DICOM file."""
    from pydicom import dcmread

    ds = dcmread(file_path)
    return ds, str(file_path)


async def main():
    from pydicom_jpeg_decoder import install_plugins

    install_plugins(remove_existing=True)

    results = await asyncio.gather(
        *[fetch_and_decode_dicom(file) for file in CONTROL_IMAGES]
    )

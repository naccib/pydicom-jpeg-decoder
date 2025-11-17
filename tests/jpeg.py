from io import BytesIO

import pytest
from httpx import AsyncClient
from pydicom import dcmread

SUPPORTED_JPEG_ENCODED_DICOM_FILES = [
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

UNSUPPORTED_12_BIT_JPEG_ENCODED_DICOM_FILES = [
    "ljdata/ds/JPEGExtended/JPEG-lossy.dcm",
    "ljdata/ds/JPEGExtended/JPEGExtended_1s_1f_u_16_12.dcm",
    "ljdata/ds/JPEGExtended/RG2_JPLY.dcm",
    "ljdata/ds/JPEGExtended/RG2_JPLY_fixed.dcm",
]


async def fetch_test_file(path: str) -> bytes:
    async with AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            f"https://github.com/pydicom/pylibjpeg-data/raw/main/{path}"
        )

        response.raise_for_status()

        assert response.content is not None, f"Failed to fetch {path}: no content"
        assert len(response.content) > 0, f"Failed to fetch {path}: empty content"

        return response.content


@pytest.mark.asyncio
@pytest.mark.parametrize("file", SUPPORTED_JPEG_ENCODED_DICOM_FILES)
async def test_should_decode_jpeg_encoded_dicom_file(file: str):
    data = await fetch_test_file(file)
    buffer = BytesIO(data)

    ds = dcmread(buffer)

    assert ds.pixel_array is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("file", UNSUPPORTED_12_BIT_JPEG_ENCODED_DICOM_FILES)
async def test_should_not_decode_12_bit_jpeg_encoded_dicom_file(file: str):
    data = await fetch_test_file(file)
    buffer = BytesIO(data)

    with pytest.raises(match=r"Unsupported\(SamplePrecision\(12\)\)"):
        ds = dcmread(buffer)
        assert ds.pixel_array is None

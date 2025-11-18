import asyncio
from io import BytesIO
from logging import basicConfig
from pathlib import Path

import numpy as np
from httpx import AsyncClient
from matplotlib import pyplot as plt

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

    # Decode local edge case files first
    print("Decoding local edge case files...")
    local_results = [
        decode_local_dicom(CLAIMS_TO_BE_RGB_BUT_IS_NOT_PATH),
        decode_local_dicom(CLAIMS_TO_BE_RGB_AND_ACTUALLY_IS_BUT_HOROS_FUCKS_IT_UP_PATH),
    ]

    # Fetch and decode all control images in parallel
    print(f"Fetching and decoding {len(CONTROL_IMAGES)} control images...")
    control_results = await asyncio.gather(
        *[fetch_and_decode_dicom(file) for file in CONTROL_IMAGES]
    )

    # Combine results: local files first, then control images
    results = local_results + list(control_results)

    # Calculate grid dimensions
    num_images = len(results)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]

    # Render each image
    for idx, (ds, file_path) in enumerate(results):
        ax = axes[idx]
        pixel_array = ds.pixel_array

        # Handle different array dimensions
        if pixel_array.ndim == 4:
            # Multi-frame: take first frame
            image = pixel_array[0, ...]
        elif pixel_array.ndim == 3:
            # Single frame with channels
            image = pixel_array[...]
        elif pixel_array.ndim == 2:
            # Grayscale
            image = pixel_array[...]
        else:
            # Fallback: take first slice
            image = pixel_array[0, ...] if pixel_array.ndim > 2 else pixel_array

        # Display the image
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        # Set title from filename and PhotometricInterpretation
        filename = Path(file_path).name
        photometric = ds.get("PhotometricInterpretation")
        photometric_str = photometric if photometric else "N/A"
        title = f"{filename}\nPI={photometric_str}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Optional, cast

import polars as pl
from cyclopts import App, Parameter
from httpx import AsyncClient
from pydicom import dcmread
from pydicom.pixels.decoders.base import JPEGLosslessSV1Decoder
from rich import print
from rich.progress import Progress

app = App()


JPEG_LOSSLESS_SV1_IMAGES = [
    "ljdata/ds/JPEGLosslessSV1/532_JPEGLossless_VOI.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEG-LL.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_1s_1f_u_08_08.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_1s_1f_u_16_16.dcm",
    "ljdata/ds/JPEGLosslessSV1/JPEGLosslessP14SV1_3s_2f_u_08_08.dcm",
    "ljdata/ds/JPEGLosslessSV1/MG1_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/RG1_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/RG2_JPLL.dcm",
    "ljdata/ds/JPEGLosslessSV1/SC_rgb_jpeg_gdcm.dcm",
]

Plugin = pl.Enum(
    ["pylibjpeg", "pydicom-jpeg-decoder"],
)

COMPARISION_DATAFRAME_SCHEMA = pl.Schema(
    {
        "image": pl.Utf8,
        "plugin": Plugin,
        "time-ms": pl.Float64,
    }
)

# MARK: Run command


@app.command
async def run(
    save_to_path: Annotated[Optional[Path], Parameter(name="save-to")] = Path(
        ".benchmark/data.csv"
    ),
) -> pl.DataFrame:
    """
    Runs the performance tests.

    Can be used either as a CLI or as a function.
    """

    data = await run_jpeg_lossless_sv1_performance_comparision()

    if save_to_path is not None:
        save_to_path.parent.mkdir(parents=True, exist_ok=True)

        data.write_csv(save_to_path)

        info(f"saved results to {save_to_path}")

    return data


async def run_jpeg_lossless_sv1_performance_comparision(
    number_of_iterations: int = 100,
) -> pl.DataFrame:
    """
    Runs the performance comparision for the JPEG Lossless SV1 images.

    Compares between the "pylibjpeg" (control) and "pydicom-jpeg-decoder" plugins.
    """

    data = []

    images = await fetch_images_in_parallel(JPEG_LOSSLESS_SV1_IMAGES)
    info("fetched images")

    total_iterations = len(images) * number_of_iterations

    assert (
        len(JPEGLosslessSV1Decoder.available_plugins) == 1
        and "pylibjpeg" in JPEGLosslessSV1Decoder.available_plugins
    ), (
        f"Expected only the 'pylibjpeg' plugin to be available and to be the only available plugin, but got {JPEGLosslessSV1Decoder.available_plugins}"
    )

    info("starting decoding using [bold]pylibjpeg[/bold]")

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Decoding with pylibjpeg...", total=total_iterations
        )

        for i in range(number_of_iterations):
            for path, image in images:
                data.append(
                    {
                        "image": path,
                        "plugin": "pylibjpeg",
                        "time-ms": decode_image(image),
                    }
                )
                progress.update(task, advance=1)

    # Switch to pydicom-jpeg-decoder plugin
    from pydicom_jpeg_decoder import install_plugins

    install_plugins(remove_existing=True)

    assert (
        len(JPEGLosslessSV1Decoder.available_plugins) == 1
        and "pydicom-jpeg-decoder" in JPEGLosslessSV1Decoder.available_plugins
    ), (
        f"Expected the 'pydicom-jpeg-decoder' plugin to be available and to be the only available plugin, but got {JPEGLosslessSV1Decoder.available_plugins}"
    )

    info("starting decoding using [bold]pydicom-jpeg-decoder[/bold]")

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Decoding with pydicom-jpeg-decoder...",
            total=total_iterations,
        )

        for _ in range(number_of_iterations):
            for path, image in images:
                data.append(
                    {
                        "image": path,
                        "plugin": "pydicom-jpeg-decoder",
                        "time-ms": decode_image(image),
                    }
                )
                progress.update(task, advance=1)

    info("decoding done")

    return pl.DataFrame(data, schema=COMPARISION_DATAFRAME_SCHEMA)


def decode_image(image: bytes) -> float:
    """
    Decodes the image for the given plugin.
    """

    buffer = BytesIO(image)
    start_time = perf_counter()

    ds = dcmread(buffer)

    # NOTE: this forces the pixel array to be decoded without any side effects (cast is essentially a no-op)
    _ = cast(Any, ds.pixel_array)

    end_time = perf_counter()

    return (end_time - start_time) * 1000


# MARK: Report command


@app.command
async def report(
    data: Annotated[Path, Parameter(name="data")] = Path(".benchmark/data.csv"),
):
    """
    Reports the performance results.
    """

    df: pl.DataFrame = pl.read_csv(data, schema=COMPARISION_DATAFRAME_SCHEMA)

    grouped = (
        df.group_by(["plugin"])
        .agg(
            pl.col("time-ms").mean().alias("time-ms-mean"),
            pl.col("time-ms").std().alias("time-ms-std"),
        )
        .sort("plugin")
    )

    # Extract values for each plugin
    pylibjpeg_row = grouped.filter(pl.col("plugin") == "pylibjpeg")
    pydicom_row = grouped.filter(pl.col("plugin") == "pydicom-jpeg-decoder")

    pylibjpeg_mean = pylibjpeg_row["time-ms-mean"][0]
    pylibjpeg_std = pylibjpeg_row["time-ms-std"][0]
    pydicom_mean = pydicom_row["time-ms-mean"][0]
    pydicom_std = pydicom_row["time-ms-std"][0]

    # Calculate delta percentage
    delta_pct = ((pydicom_mean - pylibjpeg_mean) / pylibjpeg_mean) * 100

    # Calculate individual deltas for each image/iteration pair to get SD
    deltas = []
    for image in df["image"].unique():
        pylibjpeg_times = df.filter(
            (pl.col("image") == image) & (pl.col("plugin") == "pylibjpeg")
        )["time-ms"].to_list()
        pydicom_times = df.filter(
            (pl.col("image") == image) & (pl.col("plugin") == "pydicom-jpeg-decoder")
        )["time-ms"].to_list()

        # Calculate delta for each paired measurement
        for pylibjpeg_time, pydicom_time in zip(pylibjpeg_times, pydicom_times):
            if pylibjpeg_time != 0:
                delta = ((pydicom_time - pylibjpeg_time) / pylibjpeg_time) * 100
                deltas.append(delta)

    delta_std = pl.Series(deltas).std() if deltas else 0.0

    # Format time as "mean ± std"
    pylibjpeg_time = f"{pylibjpeg_mean:.2f} (± {pylibjpeg_std:.2f})"
    pydicom_time = f"{pydicom_mean:.2f} (± {pydicom_std:.2f})"

    # Format delta with SD
    delta_str = (
        f"{delta_pct:+.1f}% (± {delta_std:.1f}%)"
        if delta_pct != 0
        else f"0.0% (± {delta_std:.1f}%)"
    )

    # Create report DataFrame
    report_df = pl.DataFrame(
        {
            "Plugin": ["pydicom-jpeg-decoder", "pylibjpeg"],
            "Time (ms)": [pydicom_time, pylibjpeg_time],
            "Delta (%)": [delta_str, "N/A"],
        }
    )

    # Render as markdown table
    columns = report_df.columns
    rows = report_df.to_dicts()

    # Calculate column widths
    col_widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            col_widths[col] = max(col_widths[col], len(str(row[col])))

    # Build markdown table
    def format_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                f"{val:<{col_widths[col]}}" for col, val in zip(columns, values)
            )
            + " |"
        )

    def format_separator() -> str:
        return "| " + " | ".join("-" * col_widths[col] for col in columns) + " |"

    lines = [
        format_row(list(columns)),
        format_separator(),
    ]

    for row in rows:
        lines.append(format_row([str(row[col]) for col in columns]))

    print("\n".join(lines))


# MARK: Utils


async def fetch_images_in_parallel(paths: list[str]) -> list[tuple[str, bytes]]:
    """
    Fetch a list of test files from the [pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) repository.
    """

    tasks = [fetch_test_file_from_pylibjpeg_data(path) for path in paths]
    return await asyncio.gather(*tasks)


async def fetch_test_file_from_pylibjpeg_data(path: str) -> tuple[str, bytes]:
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

        return path, response.content


def info(message: str):
    print(f"[bold bright_black]info: [/bold bright_black]{message}")


# MARK: Main


def main():
    app()


if __name__ == "__main__":
    main()

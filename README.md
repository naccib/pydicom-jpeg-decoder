# pydicom-jpeg-decoder

JPEG decoding for pydicom powered by pure-Rust [jpeg-decoder](https://crates.io/crates/jpeg-decoder).

## Motivation

JPEG implementations are a bit messy. After `libjpeg` release 6b in 1998, some diverging implementations containing useful but niche features were created. pydicom solves this by binding Thomas Richter's `libjpeg` library to Python and wrapping it as a pydicom plugin. 

Unfortunately, `pydicom-libjpeg` bundles a GPL-licensed libjpeg build and thus any other software that depends on it must also be GPL-licensed. This makes it difficult to use in commercial projects.

`jpeg-decoder` is a pure-Rust library that implements JPEG decoding (including JPEG Lossless and JPEG Extended) under a permissive license. This project simply wraps it using `maturin` and defines a pydicom plugin for it. We maintain a fork to include a very small feature needed to expose the color transforms of the decoded image, which is sometimes relevant for handling edge cases in pydicom.

> Why forking instead of contributing to the `jpeg-decoder` crate?
>
> According to the `jpeg-decoder` authors, the crate is in maintenance mode while they transition to `zune-jpeg` (a new, faster JPEG decoder written in Rust). The reason they still keep `jpeg-decoder` around is because it handles JPEG Lossless (precisely what we want) and only PRs related to that functionality will be accepted.

## Features
 - A JPEG decoder plugin for pydicom is available for the following transfer syntaxes: JPEG Baseline, JPEG Extended, JPEG Lossless, and JPEG Lossless SV1.

## Installation

Install it from PyPI (uv is preferred, but any other package manager will work):

```bash
uv install pydicom-jpeg-decoder
```

## Use

```python
from pydicom import dcmread
from pydicom_jpeg_decoder import install_plugins

install_plugins()

ds = dcmread("path/to/your/dicom/file.dcm")

print(f"{ds.pixel_array.shape=}")
```

## Testing

We test against the [pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) dataset of JPEG-encoded DICOM files. Simply run `pytest` to run the tests.

## Color conversion details

We cannot control what final color space `jpeg-decoder` will return. It will always return `RGB` for 3-channel images. We therefore always set the photometric interpretation to `RGB`.

## Benchmark results

We compare the performance of `pydicom-jpeg-decoder` against `pylibjpeg` on the JPEG Lossless SV1 images from the [pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) dataset. Each image is decoded 100 times for each plugin. The table below shows the results, which essentially boil down to a **mean 42% speedup over `pylibjpeg`**.

| Plugin               | Time (ms)       | Delta (%)        |
| -------------------- | --------------- | ---------------- |
| pydicom-jpeg-decoder | 44.51 (± 53.00) | -41.7% (± 33.6%) |
| pylibjpeg            | 76.35 (± 92.08) | N/A              |

You can run it yourself by running `uv run tests/performance.py run` followed by `uv run tests/performance.py report`.
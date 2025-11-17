# pydicom-jpeg-decoder

JPEG decoding for pydicom powered by pure-Rust [jpeg-decoder](https://crates.io/crates/jpeg-decoder).

## Motivation

JPEG implementations are a bit messy. After `libjpeg` release 6b, in 1998, some diverging implementations containing useful but niche features were created. pydicom solves this by binding Thomas Richter's `libjpeg` library to Python and wrapping it as a pydicom plugin. 

Unfortunately, `pydicom-libjpeg` is bundles a GPL-licensed libjpeg build and thus any other software that depends on it must also be GPL-licensed. This makes it difficult to use in commercial projects.

`jpeg-decoder` is a pure-Rust library that implements JPEG decoding (including JPEG Lossless and JPEG Extended) under a permissive license. This project simply wraps it using `maturin` and defines a pydicom plugin for it.

## Features
 - JPEG decoder plugin for pydicom is available for the following transfer syntaxes: JPEG Baseline, JPEG Extended, JPEG Lossless, and JPEG Lossless SV1.

## Installation

Right now this project is not yet available on PyPI. No wheels are published.

## Use

```python
from pydicom import dcmread
from pydicom_jpeg_decoder import decode_jpeg

ds = dcmread("path/to/your/dicom/file.dcm")

print(f"{ds.pixel_array.shape=}")
```

## Testing

We regress against the [pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) dataset of JPEG-encoded DICOM files. Simply run `pytest` to run the tests.

use pyo3::prelude::*;

/// Decodes JPEG images into a raw pixel data using `jpeg-decoder`.
#[pymodule]
mod pydicom_jpeg_decoder {
    use std::io::BufReader;

    use pyo3::prelude::*;

    /// Decodes the bytes of a JPEG image into a byte vector.
    #[pyfunction]
    fn decode_jpeg(data: &[u8]) -> PyResult<Vec<u8>> {
        let buffer = BufReader::new(data);

        let mut decoder = jpeg_decoder::Decoder::new(buffer);
        let pixel_data = decoder.decode().unwrap();

        Ok(pixel_data)
    }
}

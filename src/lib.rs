use pyo3::prelude::*;

/// Decodes JPEG images into a raw pixel data using `jpeg-decoder`.
#[pymodule]
mod pydicom_jpeg_decoder {
    use std::io::BufReader;

    use pyo3::prelude::*;

    #[pyclass]
    pub struct DecodedJpeg {
        /*
        The pixel data of the decoded JPEG image.
         */
        #[pyo3(get)]
        pub pixel_data: Vec<u8>,

        /*
        The color transform of the decoded image.
         */
        #[pyo3(get)]
        pub color_transform: Option<String>,

        /*
        The Adobe color transform of the decoded image.
         */
        #[pyo3(get)]
        pub adobe_color_transform: Option<String>,

        /*
        The color transform of the decoded image as determined by the decoder.
         */
        #[pyo3(get)]
        pub determined_color_transform: String,

        #[pyo3(get)]
        pub width: u16,

        #[pyo3(get)]
        pub height: u16,
    }

    /// Decodes the bytes of a JPEG image into a byte vector.
    #[pyfunction]
    fn decode_jpeg(data: &[u8]) -> PyResult<DecodedJpeg> {
        let buffer = BufReader::new(data);

        let mut decoder = jpeg_decoder::Decoder::new(buffer);
        let pixel_data = decoder.decode().unwrap();
        let info = decoder.info().unwrap();

        Ok(DecodedJpeg {
            pixel_data,
            color_transform: decoder
                .color_transform
                .map(|transform| color_transform_to_string(transform)),
            adobe_color_transform: decoder
                .adobe_color_transform
                .map(|transform| match transform {
                    jpeg_decoder::AdobeColorTransform::Unknown => "Unknown".to_string(),
                    jpeg_decoder::AdobeColorTransform::YCbCr => "YCbCr".to_string(),
                    jpeg_decoder::AdobeColorTransform::YCCK => "YCCK".to_string(),
                }),
            determined_color_transform: color_transform_to_string(
                decoder.determine_color_transform(),
            ),
            width: info.width,
            height: info.height,
        })
    }

    #[pyfunction]
    /// Determines the color transform of a JPEG image. Does not decode the image.
    fn determine_color_transform(data: &[u8]) -> PyResult<String> {
        let buffer = BufReader::new(data);

        let mut decoder = jpeg_decoder::Decoder::new(buffer);
        decoder.read_info().unwrap();

        let color_transform = decoder.determine_color_transform();
        Ok(color_transform_to_string(color_transform))
    }

    fn color_transform_to_string(transform: jpeg_decoder::ColorTransform) -> String {
        match transform {
            jpeg_decoder::ColorTransform::None => "None".to_string(),
            jpeg_decoder::ColorTransform::Unknown => "Unknown".to_string(),
            jpeg_decoder::ColorTransform::Grayscale => "Grayscale".to_string(),
            jpeg_decoder::ColorTransform::RGB => "RGB".to_string(),
            jpeg_decoder::ColorTransform::YCbCr => "YCbCr".to_string(),
            jpeg_decoder::ColorTransform::CMYK => "CMYK".to_string(),
            jpeg_decoder::ColorTransform::YCCK => "YCCK".to_string(),
            jpeg_decoder::ColorTransform::JcsBgYcc => "JcsBgYcc".to_string(),
            jpeg_decoder::ColorTransform::JcsBgRgb => "JcsBgRgb".to_string(),
            _ => unreachable!(),
        }
    }
}

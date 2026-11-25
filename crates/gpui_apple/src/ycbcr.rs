//! The YCbCr to RGB transform of a video surface.

use objc2_core_foundation::{CFString, CFType};
use objc2_core_video::{
    kCVImageBufferYCbCrMatrix_ITU_R_601_4, kCVImageBufferYCbCrMatrix_ITU_R_709_2,
    kCVImageBufferYCbCrMatrix_ITU_R_2020,
};

/// The YCbCr matrix of a video buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Matrix {
    Bt601,
    Bt709,
    /// BT.2020 with non-constant luminance.
    Bt2020,
}

/// The range of a buffer's 8-bit samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Levels {
    /// Luma and chroma use 0 to 255.
    Full,
    /// Luma uses 16 to 235 and chroma uses 16 to 240.
    Video,
}

impl Matrix {
    /// The luma weights of red and blue, Kr and Kb.
    fn luma_weights(self) -> (f64, f64) {
        match self {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
        }
    }
}

/// Returns the column-major matrix that maps `(y, cb, cr, 1)` to `(r, g, b, 1)`.
/// Each input is an 8-bit sample divided by 255.
///
/// The chroma center is 128/255 in both levels, so a sample with neutral
/// chroma decodes to a gray, and at full range that gray has the luma's value.
/// Only the matrix changes with the buffer's tag. Primaries and transfer are not converted, so a BT.2020 buffer or a
/// buffer with a non-sRGB transfer draws with the matrix alone.
pub(crate) fn ycbcr_transform(matrix: Matrix, levels: Levels) -> [[f32; 4]; 4] {
    let (kr, kb) = matrix.luma_weights();
    let kg = 1.0 - kr - kb;
    let (luma_scale, luma_floor, chroma_scale) = match levels {
        Levels::Full => (1.0, 0.0, 1.0),
        Levels::Video => (255.0 / 219.0, 16.0 / 255.0, 255.0 / 224.0),
    };
    let chroma_center = 128.0 / 255.0;

    let cr_to_r = 2.0 * (1.0 - kr) * chroma_scale;
    let cb_to_g = -2.0 * kb * (1.0 - kb) / kg * chroma_scale;
    let cr_to_g = -2.0 * kr * (1.0 - kr) / kg * chroma_scale;
    let cb_to_b = 2.0 * (1.0 - kb) * chroma_scale;
    let luma_bias = -luma_scale * luma_floor;

    let columns: [[f64; 4]; 4] = [
        [luma_scale, luma_scale, luma_scale, 0.0],
        [0.0, cb_to_g, cb_to_b, 0.0],
        [cr_to_r, cr_to_g, 0.0, 0.0],
        [
            luma_bias - cr_to_r * chroma_center,
            luma_bias - (cb_to_g + cr_to_g) * chroma_center,
            luma_bias - cb_to_b * chroma_center,
            1.0,
        ],
    ];
    columns.map(|column| column.map(|value| value as f32))
}

/// Returns the matrix that a buffer's `kCVImageBufferYCbCrMatrixKey`
/// attachment names. No tag, or a tag that names no matrix here, gives BT.601.
pub(crate) fn matrix_from_tag(tag: Option<&CFType>) -> Matrix {
    let Some(name) = tag.and_then(|tag| tag.downcast_ref::<CFString>()) else {
        return Matrix::Bt601;
    };
    // SAFETY: The constants are immutable strings that Core Video exports.
    let known = unsafe {
        [
            (kCVImageBufferYCbCrMatrix_ITU_R_709_2, Matrix::Bt709),
            (kCVImageBufferYCbCrMatrix_ITU_R_601_4, Matrix::Bt601),
            (kCVImageBufferYCbCrMatrix_ITU_R_2020, Matrix::Bt2020),
        ]
    };
    known
        .into_iter()
        .find(|(constant, _)| name == *constant)
        .map_or(Matrix::Bt601, |(_, matrix)| matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::CFNumber;
    use objc2_core_video::kCVImageBufferYCbCrMatrix_SMPTE_240M_1995;

    const MATRICES: [Matrix; 3] = [Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020];
    const LEVELS: [Levels; 2] = [Levels::Full, Levels::Video];

    /// The BT.601 full range matrix as JFIF writes it: coefficients rounded to
    /// four places and a chroma center of 0.5.
    const JFIF_601_LITERAL: [[f32; 4]; 4] = [
        [1.0, 1.0, 1.0, 0.0],
        [0.0, -0.3441, 1.7720, 0.0],
        [1.4020, -0.7141, 0.0, 0.0],
        [-0.7010, 0.5291, -0.8860, 1.0],
    ];

    fn decode(transform: [[f32; 4]; 4], ycbcr: [f32; 3]) -> [f32; 3] {
        let input = [ycbcr[0], ycbcr[1], ycbcr[2], 1.0];
        let mut rgb = [0.0; 3];
        for (row, value) in rgb.iter_mut().enumerate() {
            *value = (0..4)
                .map(|column| transform[column][row] * input[column])
                .sum();
        }
        rgb
    }

    fn normalized(sample: u8) -> f32 {
        sample as f32 / 255.0
    }

    /// Asserts the Cr to R, Cb to G, Cr to G and Cb to B coefficients.
    fn assert_chroma_coefficients(transform: [[f32; 4]; 4], expected: [f32; 4]) {
        let actual = [
            transform[2][0],
            transform[1][1],
            transform[2][1],
            transform[1][2],
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} differs from {expected}"
            );
        }
        assert_eq!(transform[1][0], 0.0);
        assert_eq!(transform[2][2], 0.0);
    }

    #[test]
    fn the_601_full_range_coefficients_equal_the_jfif_values_within_5e_5() {
        let transform = ycbcr_transform(Matrix::Bt601, Levels::Full);
        for column in 0..3 {
            for row in 0..4 {
                let (actual, jfif) = (transform[column][row], JFIF_601_LITERAL[column][row]);
                assert!(
                    (actual - jfif).abs() < 5e-5,
                    "column {column} row {row}: {actual} differs from {jfif}"
                );
            }
        }
    }

    #[test]
    fn the_601_full_range_bias_is_the_jfif_bias_moved_to_the_128_255_center() {
        let transform = ycbcr_transform(Matrix::Bt601, Levels::Full);
        let center_ratio = (128.0 / 255.0) / 0.5;
        for row in 0..3 {
            let (actual, jfif) = (transform[3][row], JFIF_601_LITERAL[3][row]);
            assert_ne!(actual, jfif);
            assert!(
                (actual - jfif * center_ratio).abs() < 5e-5,
                "row {row}: {actual} is not {jfif} moved to the 128/255 center"
            );
        }
        assert_eq!(transform[3][3], 1.0);
    }

    #[test]
    fn the_709_full_range_transform_has_the_709_coefficients() {
        let transform = ycbcr_transform(Matrix::Bt709, Levels::Full);
        assert_eq!(transform[0], [1.0, 1.0, 1.0, 0.0]);
        assert_chroma_coefficients(transform, [1.5748, -0.187324, -0.468124, 1.8556]);
    }

    #[test]
    fn the_2020_transform_has_the_2020_coefficients() {
        let transform = ycbcr_transform(Matrix::Bt2020, Levels::Full);
        assert_eq!(transform[0], [1.0, 1.0, 1.0, 0.0]);
        assert_chroma_coefficients(transform, [1.4746, -0.164553, -0.571353, 1.8814]);
    }

    #[test]
    fn the_video_range_transform_has_the_scaled_coefficients() {
        let luma_scale = 255.0 / 219.0;
        for (matrix, expected) in [
            (Matrix::Bt601, [1.596027, -0.391762, -0.812968, 2.017232]),
            (Matrix::Bt709, [1.792741, -0.213249, -0.532909, 2.112402]),
        ] {
            let transform = ycbcr_transform(matrix, Levels::Video);
            for row in 0..3 {
                assert!((transform[0][row] - luma_scale).abs() < 1e-6);
            }
            assert_eq!(transform[0][3], 0.0);
            assert_chroma_coefficients(transform, expected);
        }
    }

    #[test]
    fn a_neutral_gray_decodes_to_the_same_gray_exactly_in_every_matrix_and_level() {
        let center = normalized(128);
        for matrix in MATRICES {
            for levels in LEVELS {
                let transform = ycbcr_transform(matrix, levels);
                for luma in (16..=235).step_by(3).chain([128]) {
                    let gray = match levels {
                        Levels::Full => normalized(luma),
                        Levels::Video => (luma as f32 - 16.0) / 219.0,
                    };
                    let rgb = decode(transform, [normalized(luma), center, center]);
                    for value in rgb {
                        assert!(
                            (value - gray).abs() < 1e-6,
                            "{matrix:?} {levels:?} luma {luma}: {rgb:?} is not {gray}"
                        );
                    }
                    if levels == Levels::Full {
                        assert!(
                            rgb.iter()
                                .all(|value| (value * 255.0).round() == luma as f32)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_709_video_range_triple_decodes_to_its_color() {
        let transform = ycbcr_transform(Matrix::Bt709, Levels::Video);
        for (ycbcr, color) in [
            ([63, 102, 240], [1.0, 0.0, 0.0]),
            ([173, 42, 26], [0.0, 1.0, 0.0]),
            ([32, 240, 118], [0.0, 0.0, 1.0]),
            ([235, 128, 128], [1.0, 1.0, 1.0]),
            ([16, 128, 128], [0.0, 0.0, 0.0]),
        ] {
            let rgb = decode(transform, ycbcr.map(normalized));
            for (value, expected) in rgb.into_iter().zip(color) {
                assert!(
                    (value - expected).abs() < 0.01,
                    "{ycbcr:?} decodes to {rgb:?}, not {color:?}"
                );
            }
        }
    }

    #[test]
    fn row_three_is_zero_zero_zero_one() {
        for matrix in MATRICES {
            for levels in LEVELS {
                let transform = ycbcr_transform(matrix, levels);
                let row_three = transform.map(|column| column[3]);
                assert_eq!(row_three, [0.0, 0.0, 0.0, 1.0], "{matrix:?} {levels:?}");
            }
        }
    }

    #[test]
    fn a_missing_tag_gives_601() {
        assert_eq!(matrix_from_tag(None), Matrix::Bt601);
    }

    #[test]
    fn the_709_2_601_4_and_2020_tags_give_their_matrices() {
        // SAFETY: The constants are immutable strings that Core Video exports.
        let tags = unsafe {
            [
                (kCVImageBufferYCbCrMatrix_ITU_R_709_2, Matrix::Bt709),
                (kCVImageBufferYCbCrMatrix_ITU_R_601_4, Matrix::Bt601),
                (kCVImageBufferYCbCrMatrix_ITU_R_2020, Matrix::Bt2020),
            ]
        };
        for (tag, matrix) in tags {
            assert_eq!(matrix_from_tag(Some(tag)), matrix);
            // A copy of the tag compares by value.
            let copy = CFString::from_str(&tag.to_string());
            assert_eq!(matrix_from_tag(Some(&copy)), matrix);
        }
    }

    #[test]
    fn an_unknown_string_gives_601() {
        // SAFETY: The constant is an immutable string that Core Video exports.
        let smpte_240m = unsafe { kCVImageBufferYCbCrMatrix_SMPTE_240M_1995 };
        assert_eq!(matrix_from_tag(Some(smpte_240m)), Matrix::Bt601);
        let unknown = CFString::from_str("ITU_R_709_3");
        assert_eq!(matrix_from_tag(Some(&unknown)), Matrix::Bt601);
        let empty = CFString::from_str("");
        assert_eq!(matrix_from_tag(Some(&empty)), Matrix::Bt601);
    }

    #[test]
    fn a_value_that_is_not_a_string_gives_601() {
        let number = CFNumber::new_i32(709);
        assert_eq!(matrix_from_tag(Some(&number)), Matrix::Bt601);
    }
}

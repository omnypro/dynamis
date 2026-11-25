//! Video surfaces of known YCbCr samples, for a check by eye. Each row is one
//! buffer, and the squares to its right are the RGB colors that its bars
//! should show.
//!
//! The first three rows are `420f` buffers and the last three are `420v`
//! buffers. In each format the rows are tagged BT.601, tagged BT.709 and
//! untagged, in that order, and the untagged row should match the BT.601 row.
//! Every buffer holds the same samples: BT.709 video range red, green, blue,
//! white and black.

#[cfg(target_os = "macos")]
mod surface_colors {
    use std::{ptr, slice};

    use gpui::{
        App, Bounds, Context, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
        surface,
    };
    use gpui_platform::application;
    use objc2_core_foundation::{CFDictionary, CFRetained, CFString, CFType};
    use objc2_core_video::{
        CVAttachmentMode, CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane,
        CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVImageBufferYCbCrMatrix_ITU_R_601_4,
        kCVImageBufferYCbCrMatrix_ITU_R_709_2, kCVImageBufferYCbCrMatrixKey,
        kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVReturnSuccess,
    };

    /// Y, Cb and Cr of each bar.
    const SAMPLES: [[u8; 3]; 5] = [
        [63, 102, 240],
        [173, 42, 26],
        [32, 240, 118],
        [235, 128, 128],
        [16, 128, 128],
    ];
    const BAR_SIZE: usize = 48;

    #[derive(Clone, Copy)]
    enum Matrix {
        Bt601,
        Bt709,
    }

    impl Matrix {
        /// The luma weights of red and blue, Kr and Kb.
        fn luma_weights(self) -> (f32, f32) {
            match self {
                Matrix::Bt601 => (0.299, 0.114),
                Matrix::Bt709 => (0.2126, 0.0722),
            }
        }
    }

    struct Row {
        buffer: CFRetained<CVPixelBuffer>,
        colors: [u32; 5],
    }

    struct SurfaceColors {
        rows: Vec<Row>,
    }

    impl SurfaceColors {
        fn new() -> Self {
            let mut rows = Vec::new();
            for video_range in [false, true] {
                for tag in [Some(Matrix::Bt601), Some(Matrix::Bt709), None] {
                    // A buffer with no tag decodes as BT.601.
                    let matrix = tag.unwrap_or(Matrix::Bt601);
                    rows.push(Row {
                        buffer: bars_buffer(video_range, tag),
                        colors: SAMPLES.map(|sample| expected_color(sample, matrix, video_range)),
                    });
                }
            }
            Self { rows }
        }
    }

    impl Render for SurfaceColors {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bar = px(BAR_SIZE as f32);
            div()
                .size_full()
                .flex()
                .flex_col()
                .gap_2()
                .p_4()
                .bg(rgb(0x808080))
                .children(self.rows.iter().map(|row| {
                    div()
                        .flex()
                        .gap_4()
                        .child(
                            surface(row.buffer.clone())
                                .w(bar * SAMPLES.len() as f32)
                                .h(bar),
                        )
                        .child(
                            div()
                                .flex()
                                .children(row.colors.map(|color| div().size(bar).bg(rgb(color)))),
                        )
                }))
        }
    }

    /// Returns the color that a sample shows, as `0xRRGGBB`, from the luma
    /// equation `Y = Kr R + Kg G + Kb B` and a chroma center of 128.
    fn expected_color([y, cb, cr]: [u8; 3], matrix: Matrix, video_range: bool) -> u32 {
        let (y, cb, cr) = (y as f32, cb as f32 - 128.0, cr as f32 - 128.0);
        let (y, cb, cr) = if video_range {
            ((y - 16.0) / 219.0, cb / 224.0, cr / 224.0)
        } else {
            (y / 255.0, cb / 255.0, cr / 255.0)
        };
        let (kr, kb) = matrix.luma_weights();
        let red = y + 2.0 * (1.0 - kr) * cr;
        let blue = y + 2.0 * (1.0 - kb) * cb;
        let green = (y - kr * red - kb * blue) / (1.0 - kr - kb);
        [red, green, blue]
            .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u32)
            .into_iter()
            .fold(0, |color, channel| (color << 8) | channel)
    }

    /// Returns a buffer of one bar per sample, tagged with `tag` when one is
    /// given.
    fn bars_buffer(video_range: bool, tag: Option<Matrix>) -> CFRetained<CVPixelBuffer> {
        let pixel_format = if video_range {
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        } else {
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        };
        // SAFETY: The key is an immutable string that Core Video exports.
        let surface_properties_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        // A Metal texture of a plane needs a buffer with an IOSurface.
        let attributes = CFDictionary::from_slices(
            &[surface_properties_key],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut buffer = ptr::null_mut();
        // SAFETY: The attributes dictionary is live and the output pointer is writable.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                BAR_SIZE * SAMPLES.len(),
                BAR_SIZE,
                pixel_format,
                Some(attributes.as_opaque()),
                ptr::NonNull::from(&mut buffer),
            )
        };
        assert_eq!(status, kCVReturnSuccess);
        // SAFETY: The successful create call transfers ownership of this buffer.
        let buffer = unsafe { CFRetained::from_raw(ptr::NonNull::new(buffer).unwrap()) };

        // SAFETY: The base address stays valid while the buffer is locked, and each row
        // write stays within the plane's width and bytes per row.
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            // Plane 0 holds one Y byte per pixel and plane 1 one Cb, Cr pair per
            // two by two pixels.
            for (plane, bytes_per_sample) in [(0, 1), (1, 2)] {
                let base = CVPixelBufferGetBaseAddressOfPlane(&buffer, plane).cast::<u8>();
                let bytes_per_row = CVPixelBufferGetBytesPerRowOfPlane(&buffer, plane);
                let width = CVPixelBufferGetWidthOfPlane(&buffer, plane);
                let bar_width = width / SAMPLES.len();
                for row in 0..CVPixelBufferGetHeightOfPlane(&buffer, plane) {
                    let row = slice::from_raw_parts_mut(
                        base.add(row * bytes_per_row),
                        width * bytes_per_sample,
                    );
                    for (column, sample) in row.chunks_exact_mut(bytes_per_sample).enumerate() {
                        let [y, cb, cr] = SAMPLES[column / bar_width];
                        if plane == 0 {
                            sample.copy_from_slice(&[y]);
                        } else {
                            sample.copy_from_slice(&[cb, cr]);
                        }
                    }
                }
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        if let Some(matrix) = tag {
            // SAFETY: The constants are immutable strings that Core Video exports, and the
            // matrix key takes a string value.
            unsafe {
                let value: &CFString = match matrix {
                    Matrix::Bt601 => kCVImageBufferYCbCrMatrix_ITU_R_601_4,
                    Matrix::Bt709 => kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                };
                buffer.set_attachment(
                    kCVImageBufferYCbCrMatrixKey,
                    value,
                    CVAttachmentMode::ShouldPropagate,
                );
            }
        }
        buffer
    }

    pub fn run() {
        application().run(|cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(560.), px(400.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(|_| SurfaceColors::new()),
            )
            .unwrap();
            cx.activate(true);
        });
    }
}

#[cfg(target_os = "macos")]
fn main() {
    surface_colors::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {}

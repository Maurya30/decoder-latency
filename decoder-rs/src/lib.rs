//! The 8-bit CNN speech decoder, one 20 ms row at a time.
//!
//! This is the same model as `SmallCNN` in model.py after quantize.py has turned its
//! weights into bytes, and it must give exactly the same integers as `int_forward` there.
//!
//! What "ready for a microcontroller" means here:
//!   - `no_std`: nothing from the operating system. No files, no clock, no threads.
//!   - no heap: nothing is allocated while it runs. All working memory is the fixed-size
//!     `Decoder` struct, so the RAM it needs is known when it is compiled.
//!   - the weights are borrowed, never copied. On a microcontroller they stay in flash
//!     (the chip's read-only storage) and only `Decoder` lives in RAM.
//!   - integers only, apart from scaling the incoming float features to bytes.
//!
//! How the live loop saves work. One output step looks at 19 rows, and a new step is due
//! every 4 rows. The first layer turns those 19 rows into 7 columns, and 5 of the 7 were
//! already worked out for the step before. So the decoder keeps the newest 7 rows and the
//! newest 7 first-layer columns, and does only the new work as each row arrives:
//!
//!   every 2nd row   one new first-layer column (conv1)
//!   every 4th row   the second layer and the 41 phoneme scores (conv2, head)
//!
//! `C` is the model's width: the channels in its two hidden layers (256 in the README's CNN).

#![no_std]
#![forbid(unsafe_code)]

/// Features in one 20 ms row: 256 electrodes x 2.
pub const FEATURES: usize = 512;
/// Phoneme classes, including the CTC blank at 0.
pub const CLASSES: usize = 41;
/// Rows one output step looks at (380 ms).
pub const WINDOW: usize = 19;
/// New rows between one output step and the next (80 ms).
pub const HOP: usize = 4;

const KERNEL: usize = 7; // both conv layers look at 7 neighbours
const STRIDE: usize = 2; // and both move 2 at a time
const SHARED: u8 = (KERNEL - STRIDE) as u8; // so each sum shares 5 of its 7 inputs with the one before
const HEADER: usize = 32;

/// Why a model file was turned down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelError {
    /// Not a file written by quantize.py.
    NotAModel,
    /// A model, but for a different width than this decoder was compiled for.
    WrongShape,
    /// Shorter or longer than its header says.
    WrongLength,
}

/// The model file, as written by quantize.py's `save_int8`, split into its arrays.
///
/// Every field is a slice into the caller's bytes. Nothing is copied.
pub struct Model<'a, const C: usize> {
    in_inv_scale: f32,
    shift1: u32,
    shift2: u32,
    w1: &'a [u8], // [C][7][512] signed bytes
    b1: &'a [u8], // [C] i32
    m1: &'a [u8], // [C] i32
    w2: &'a [u8], // [C][7][C] signed bytes
    b2: &'a [u8],
    m2: &'a [u8],
    w3: &'a [u8], // [41][C] signed bytes
    b3: &'a [u8], // [41] i32
}

fn u32_at(bytes: &[u8], index: usize) -> u32 {
    let at = 4 * index;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn i32_at(bytes: &[u8], index: usize) -> i32 {
    u32_at(bytes, index) as i32
}

impl<'a, const C: usize> Model<'a, C> {
    /// How many bytes a model of this width takes.
    pub const BYTES: usize =
        HEADER + C * KERNEL * FEATURES + C * KERNEL * C + CLASSES * C + 4 * (4 * C + CLASSES);

    /// Check the header and split the file into its arrays.
    pub fn parse(file: &'a [u8]) -> Result<Self, ModelError> {
        if file.len() < HEADER || &file[..4] != b"DLQ1" {
            return Err(ModelError::NotAModel);
        }
        let head = &file[4..HEADER];
        let shape = [u32_at(head, 0), u32_at(head, 1), u32_at(head, 2), u32_at(head, 3)];
        if shape != [FEATURES as u32, C as u32, CLASSES as u32, KERNEL as u32] {
            return Err(ModelError::WrongShape);
        }
        let (shift1, shift2) = (u32_at(head, 4), u32_at(head, 5));
        if file.len() != Self::BYTES || !(1..=62).contains(&shift1) || !(1..=62).contains(&shift2) {
            return Err(ModelError::WrongLength);
        }
        let in_inv_scale = f32::from_bits(u32_at(head, 6));
        let rest = &file[HEADER..];
        let (w1, rest) = rest.split_at(C * KERNEL * FEATURES);
        let (b1, rest) = rest.split_at(4 * C);
        let (m1, rest) = rest.split_at(4 * C);
        let (w2, rest) = rest.split_at(C * KERNEL * C);
        let (b2, rest) = rest.split_at(4 * C);
        let (m2, rest) = rest.split_at(4 * C);
        let (w3, b3) = rest.split_at(CLASSES * C);
        Ok(Model { in_inv_scale, shift1, shift2, w1, b1, m1, w2, b2, m2, w3, b3 })
    }

    /// Scale one row of float features to signed bytes, the way this model expects them.
    pub fn scale_row(&self, features: &[f32; FEATURES], row: &mut [i8; FEATURES]) {
        for (byte, &feature) in row.iter_mut().zip(features) {
            *byte = to_byte(feature, self.in_inv_scale);
        }
    }
}

/// Scale one float feature to a signed byte: nearest whole number, halves away from zero,
/// kept within +-127. The same sums, in the same order, as `quantize_rows` in quantize.py.
fn to_byte(feature: f32, inv_scale: f32) -> i8 {
    let scaled = feature * inv_scale;
    let nudged = if scaled >= 0.0 { scaled + 0.5 } else { scaled - 0.5 };
    (nudged as i32).clamp(-127, 127) as i8 // `as i32` drops the fraction
}

/// Sum of weight x input over one stretch of memory. Inputs are signed bytes (feature rows).
fn dot_signed(weights: &[u8], inputs: &[i8]) -> i32 {
    weights.iter().zip(inputs).map(|(&w, &x)| (w as i8 as i32) * (x as i32)).sum()
}

/// The same, for inputs that are 0..255 (a hidden layer's outputs).
fn dot_unsigned(weights: &[u8], inputs: &[u8]) -> i32 {
    weights.iter().zip(inputs).map(|(&w, &x)| (w as i8 as i32) * (x as i32)).sum()
}

/// ReLU, then bring a 32-bit sum down to 0..255: multiply by m, add half, shift right, cap.
/// quantize.py proves the sums fit in 32 bits; the product needs 64.
fn requantize(sum: i32, m: i32, shift: u32) -> u8 {
    let scaled = (sum.max(0) as i64 * m as i64 + (1i64 << (shift - 1))) >> shift;
    scaled.min(255) as u8
}

/// All the memory the decoder uses while it runs.
pub struct Decoder<const C: usize> {
    rows: [[i8; FEATURES]; KERNEL], // the newest 7 input rows, as a ring
    cols: [[u8; C]; KERNEL],        // the newest 7 first-layer columns, as a ring
    hidden: [u8; C],                // the second layer's output for the current step
    scores: [i32; CLASSES],         // the current step's phoneme scores
    row_at: u8,                     // where the next row goes; once the ring is full, also the oldest row
    col_at: u8,                     // the same for columns
    rows_ready: u8,                 // how many of the next column's 7 rows have arrived
    cols_ready: u8,                 // how many of the next output step's 7 columns exist
}

impl<const C: usize> Default for Decoder<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize> Decoder<C> {
    /// A decoder that has seen no rows. `const`, so it can be a `static` on a microcontroller.
    /// Every field starts at zero, so that static costs no flash: the startup code just zeroes it.
    pub const fn new() -> Self {
        Decoder {
            rows: [[0; FEATURES]; KERNEL],
            cols: [[0; C]; KERNEL],
            hidden: [0; C],
            scores: [0; CLASSES],
            row_at: 0,
            col_at: 0,
            rows_ready: 0,
            cols_ready: 0,
        }
    }

    /// Forget every row seen so far (the start of a new sentence).
    pub fn reset(&mut self) {
        self.row_at = 0;
        self.col_at = 0;
        self.rows_ready = 0;
        self.cols_ready = 0;
    }

    /// One new row of float features, as the dataset stores them. Returns the 41 phoneme
    /// scores when this row completes an output step (the 19th row, then every 4th).
    pub fn push_features(&mut self, model: &Model<C>, features: &[f32; FEATURES]) -> Option<&[i32; CLASSES]> {
        model.scale_row(features, &mut self.rows[self.row_at as usize]);
        self.after_row(model)
    }

    /// One new row that is already scaled to signed bytes.
    pub fn push_row(&mut self, model: &Model<C>, row: &[i8; FEATURES]) -> Option<&[i32; CLASSES]> {
        self.rows[self.row_at as usize] = *row;
        self.after_row(model)
    }

    /// A whole 19-row window worked out from nothing: all 7 first-layer columns, not just
    /// the 2 new ones. Same answer as the live loop, about three times the work. It is here
    /// to measure what keeping the columns saves.
    pub fn step_from_scratch(&mut self, model: &Model<C>, window: &[[i8; FEATURES]; WINDOW]) -> &[i32; CLASSES] {
        self.reset();
        for row in window {
            self.push_row(model, row);
        }
        &self.scores
    }

    fn after_row(&mut self, model: &Model<C>) -> Option<&[i32; CLASSES]> {
        self.row_at = (self.row_at + 1) % KERNEL as u8;
        self.rows_ready += 1;
        if (self.rows_ready as usize) < KERNEL {
            return None;
        }
        self.rows_ready = SHARED;
        self.conv1_column(model);

        self.cols_ready += 1;
        if (self.cols_ready as usize) < KERNEL {
            return None;
        }
        self.cols_ready = SHARED;
        self.conv2_and_head(model);
        Some(&self.scores)
    }

    /// conv1 for the newest 7 rows: one new column of C bytes.
    fn conv1_column(&mut self, model: &Model<C>) {
        let oldest = self.row_at as usize;
        let rows = &self.rows;
        let column = &mut self.cols[self.col_at as usize];
        for (c, (out, weights)) in column.iter_mut().zip(model.w1.chunks_exact(KERNEL * FEATURES)).enumerate() {
            let mut sum = i32_at(model.b1, c);
            for (tap, w) in weights.chunks_exact(FEATURES).enumerate() {
                sum += dot_signed(w, &rows[(oldest + tap) % KERNEL]);
            }
            *out = requantize(sum, i32_at(model.m1, c), model.shift1);
        }
        self.col_at = (self.col_at + 1) % KERNEL as u8;
    }

    /// conv2 over the newest 7 columns, then the head: one row of phoneme scores.
    fn conv2_and_head(&mut self, model: &Model<C>) {
        let oldest = self.col_at as usize;
        let cols = &self.cols;
        for (c, (out, weights)) in self.hidden.iter_mut().zip(model.w2.chunks_exact(KERNEL * C)).enumerate() {
            let mut sum = i32_at(model.b2, c);
            for (tap, w) in weights.chunks_exact(C).enumerate() {
                sum += dot_unsigned(w, &cols[(oldest + tap) % KERNEL]);
            }
            *out = requantize(sum, i32_at(model.m2, c), model.shift2);
        }
        for (c, (score, w)) in self.scores.iter_mut().zip(model.w3.chunks_exact(C)).enumerate() {
            *score = i32_at(model.b3, c) + dot_unsigned(w, &self.hidden);
        }
    }
}

/// The best-scoring class. Ties go to the lower class number, like numpy's argmax.
pub fn best_class(scores: &[i32; CLASSES]) -> u8 {
    let mut best = 0;
    for (class, &score) in scores.iter().enumerate() {
        if score > scores[best] {
            best = class;
        }
    }
    best as u8
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    const C: usize = 16;

    /// A model file of the right shape with every weight zero.
    fn blank_model() -> Vec<u8> {
        let mut file = std::vec![0u8; Model::<C>::BYTES];
        file[..4].copy_from_slice(b"DLQ1");
        let header = [FEATURES as u32, C as u32, CLASSES as u32, KERNEL as u32, 20, 20, 12.7f32.to_bits()];
        for (i, value) in header.iter().enumerate() {
            file[4 + 4 * i..8 + 4 * i].copy_from_slice(&value.to_le_bytes());
        }
        file
    }

    #[test]
    fn features_round_to_the_nearest_byte() {
        let bytes: Vec<i8> = [0.0, 0.03, 0.04, -0.04, 10.0, -10.0, 99.0, -99.0].iter().map(|&f| to_byte(f, 12.7)).collect();
        assert_eq!(bytes, [0, 0, 1, -1, 127, -127, 127, -127]);
        assert_eq!([to_byte(0.5, 1.0), to_byte(-0.5, 1.0), to_byte(1.5, 1.0)], [1, -1, 2]);
    }

    #[test]
    fn sums_come_down_to_one_byte() {
        let m = 1 << 20; // with a shift of 30 this divides by 1024
        let bytes: Vec<u8> = [-5, 0, 1, 511, 512, 1536, 1 << 20, 1 << 30].iter().map(|&sum| requantize(sum, m, 30)).collect();
        assert_eq!(bytes, [0, 0, 0, 0, 1, 2, 255, 255]);
    }

    #[test]
    fn a_step_comes_out_on_row_19_and_then_every_4th() {
        let file = blank_model();
        let model = Model::<C>::parse(&file).unwrap();
        let mut decoder = Decoder::<C>::new();
        let row = [0i8; FEATURES];
        for n in 1..=40 {
            let due = n >= WINDOW && (n - WINDOW) % HOP == 0;
            assert_eq!(decoder.push_row(&model, &row).is_some(), due, "row {n}");
        }
        decoder.reset();
        assert!((1..WINDOW).all(|_| decoder.push_row(&model, &row).is_none()), "reset should start a new window");
        assert!(decoder.push_row(&model, &row).is_some());
    }

    #[test]
    fn a_wrong_file_is_turned_down() {
        let mut file = blank_model();
        assert_eq!(Model::<C>::parse(&file[..file.len() - 1]).err(), Some(ModelError::WrongLength));
        assert_eq!(Model::<32>::parse(&file).err(), Some(ModelError::WrongShape));
        file[0] = b'X';
        assert_eq!(Model::<C>::parse(&file).err(), Some(ModelError::NotAModel));
    }

    #[test]
    fn ties_go_to_the_lower_class() {
        let mut scores = [0i32; CLASSES];
        assert_eq!(best_class(&scores), 0);
        scores[7] = 3;
        scores[30] = 3;
        assert_eq!(best_class(&scores), 7);
    }
}

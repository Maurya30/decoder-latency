//! The decoder as a bare-metal ARM Cortex-M image, with a self-test.
//!
//! "Bare-metal" means there is no operating system underneath: the processor starts at the
//! reset vector, cortex-m-rt zeroes the RAM and calls `main`, and that is the whole
//! program. Building it answers two questions the laptop cannot:
//!
//!   - how much flash and RAM does the decoder take? (read from the linked image)
//!   - does the ARM build give the same integers as Python? (run in an emulator)
//!
//! The self-test pushes made-up feature rows through the decoder and prints a checksum of
//! every score. firmware_check.py makes the same rows, runs them through quantize.py's
//! int_forward and compares the checksum.
//!
//! An emulator shows that the image runs and what it computes. It says nothing about how
//! long a step takes on a real chip, so nothing here measures time.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use core::ptr::{addr_of, addr_of_mut};

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use decoder::{Decoder, Model, FEATURES, HOP, WINDOW};

/// The width of the model this image is built for.
const WIDTH: usize = 256;
/// Output steps in the self-test.
const STEPS: usize = 200;

/// The model file. The linker puts it in flash, and it is read from there: never copied to RAM.
static MODEL: &[u8] = include_bytes!("../../../runs/cnn_int8.bin");

// All of the decoder's working memory, and one row of incoming features. As statics they
// sit in RAM at fixed addresses, and the build says exactly how big they are.
static mut DECODER: Decoder<WIDTH> = Decoder::new();
static mut ROW: [f32; FEATURES] = [0.0; FEATURES];

// The two ends of the stack, from cortex-m-rt's linker script. The stack starts at the top
// of RAM and grows down towards the statics.
extern "C" {
    static mut _stack_end: u32;
    static _stack_start: u32;
}

const PAINT: u32 = 0xC0FF_EE11;

/// Fill the part of the stack that is not in use yet with a known pattern. Whatever no
/// longer holds the pattern at the end has been used.
fn paint_stack() {
    let mut at = addr_of_mut!(_stack_end);
    // Everything below the stack pointer is free (no interrupts are on); 16 bytes of margin anyway.
    let stop = (cortex_m::register::msp::read() - 16) as *mut u32;
    while at < stop {
        // SAFETY: between the statics and the live stack, which nothing else uses.
        unsafe {
            at.write_volatile(PAINT);
            at = at.add(1);
        }
    }
}

/// The deepest the stack has reached since power-on, in bytes.
fn stack_used() -> u32 {
    let mut at = addr_of!(_stack_end);
    let top = addr_of!(_stack_start) as u32;
    // SAFETY: only reads RAM between the two ends of the stack.
    while (at as u32) < top && unsafe { at.read_volatile() } == PAINT {
        at = unsafe { at.add(1) };
    }
    top - at as u32
}

/// Made-up feature values between -2 and 2, the same on every run.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) % 257) as f32 / 64.0 - 2.0
    }
}

/// Fold one score into the checksum (FNV-1a over its four bytes).
fn mix(mut sum: u32, score: i32) -> u32 {
    for byte in score.to_le_bytes() {
        sum = (sum ^ byte as u32).wrapping_mul(16_777_619);
    }
    sum
}

/// Print "label number" through the emulator or debugger.
fn say(out: &mut hio::HostStream, label: &[u8], number: u32) {
    let mut digits = [0u8; 10];
    let (mut at, mut left) = (digits.len(), number);
    loop {
        at -= 1;
        digits[at] = b'0' + (left % 10) as u8;
        left /= 10;
        if left == 0 {
            break;
        }
    }
    let _ = out.write_all(label);
    let _ = out.write_all(&digits[at..]);
    let _ = out.write_all(b"\n");
}

fn stop(ok: bool) -> ! {
    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}

#[entry]
fn main() -> ! {
    paint_stack();
    // SAFETY: main runs once, and nothing else touches these two statics.
    let (decoder, row) = unsafe { (&mut *addr_of_mut!(DECODER), &mut *addr_of_mut!(ROW)) };
    let Ok(mut out) = hio::hstdout() else { stop(false) };
    let Ok(model) = Model::<WIDTH>::parse(MODEL) else {
        let _ = out.write_all(b"the model file was turned down\n");
        stop(false)
    };

    let mut noise = Noise(1);
    let (mut steps, mut checksum) = (0u32, 2_166_136_261u32);
    for _ in 0..WINDOW + (STEPS - 1) * HOP {
        for feature in row.iter_mut() {
            *feature = noise.next();
        }
        if let Some(scores) = decoder.push_features(&model, row) {
            for &score in scores {
                checksum = mix(checksum, score);
            }
            steps += 1;
        }
    }

    say(&mut out, b"steps ", steps);
    say(&mut out, b"checksum ", checksum);
    say(&mut out, b"stack_bytes ", stack_used());
    say(&mut out, b"decoder_bytes ", core::mem::size_of::<Decoder<WIDTH>>() as u32);
    stop(steps as usize == STEPS)
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    stop(false)
}

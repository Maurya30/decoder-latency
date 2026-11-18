//! The laptop test bench around the decoder. rust_check.py drives it.
//!
//!     host check <model.bin>                        sentences in on stdin, scores out on stdout
//!     host time  <model.bin> <steps> <warmup> <paced: 0 or 1>   rows in on stdin, step times out as JSON
//!
//! This file uses std (files, a clock, a heap for the test data). The decoder itself, in
//! lib.rs, uses none of that: everything it needs is the `Decoder` struct declared here.
//!
//! stdin is little-endian: a u32 row count, then that many rows of 512 f32 features.
//! `check` reads sentences until a row count of 0 and answers each with a u32 step count
//! and that many rows of 41 i32 scores.

use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

use decoder::{Decoder, Model, CLASSES, FEATURES, HOP, WINDOW};

const ROW_MS: u64 = 20;

fn read_rows(input: &mut impl Read) -> io::Result<Vec<[f32; FEATURES]>> {
    let mut count = [0u8; 4];
    input.read_exact(&mut count)?;
    let mut bytes = vec![0u8; u32::from_le_bytes(count) as usize * FEATURES * 4];
    input.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(FEATURES * 4)
        .map(|row| core::array::from_fn(|i| f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap())))
        .collect())
}

/// Every sentence through the live loop. Each step is also worked out from scratch, and
/// the two must agree exactly.
fn check<const C: usize>(file: &[u8]) -> io::Result<()> {
    let model = Model::<C>::parse(file).expect("model file");
    let mut live = Box::new(Decoder::<C>::new());
    let mut scratch = Box::new(Decoder::<C>::new());
    let (mut input, mut output) = (io::stdin().lock(), io::BufWriter::new(io::stdout().lock()));
    loop {
        let rows = read_rows(&mut input)?;
        if rows.is_empty() {
            return output.flush();
        }
        let mut bytes = vec![[0i8; FEATURES]; rows.len()];
        let mut scores: Vec<[i32; CLASSES]> = Vec::new();
        live.reset();
        for (i, row) in rows.iter().enumerate() {
            model.scale_row(row, &mut bytes[i]);
            if let Some(step) = live.push_features(&model, row) {
                let window: &[[i8; FEATURES]; WINDOW] = bytes[i + 1 - WINDOW..=i].try_into().unwrap();
                assert!(step == scratch.step_from_scratch(&model, window), "live loop and from-scratch step disagree");
                scores.push(*step);
            }
        }
        output.write_all(&(scores.len() as u32).to_le_bytes())?;
        for step in &scores {
            for score in step {
                output.write_all(&score.to_le_bytes())?;
            }
        }
        output.flush()?; // the caller waits for each sentence's answer before sending the next
    }
}

fn wait_until(moment: Instant) {
    let now = Instant::now();
    if moment > now {
        std::thread::sleep(moment - now);
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Time one step two ways. "kept" is the live loop: 4 new rows in, scores out, with the
/// first layer's older columns kept from earlier steps. "redone" scales the same 4 rows
/// and then works the whole 19-row window out from scratch.
///
/// Paced means each row is held back until its real arrival time, 20 ms after the one
/// before, so the processor idles in between as it would on a real stream. A step's time
/// is then the sum of the work done for its 4 rows.
fn time<const C: usize>(file: &[u8], steps: usize, warmup: usize, paced: bool) -> io::Result<()> {
    let model = Model::<C>::parse(file).expect("model file");
    let rows = read_rows(&mut io::stdin().lock())?;
    let total = steps + warmup;
    assert!(rows.len() >= WINDOW + total * HOP, "not enough rows for that many steps");
    let lead = WINDOW - HOP; // rows that go in before the first timed step

    let mut decoder = Box::new(Decoder::<C>::new());
    let mut kept = Vec::with_capacity(total);
    for row in &rows[..lead] {
        decoder.push_features(&model, row);
    }
    let start = Instant::now();
    for step in 0..total {
        let mut spent = Duration::ZERO;
        for i in lead + step * HOP..lead + (step + 1) * HOP {
            if paced {
                wait_until(start + Duration::from_millis(ROW_MS * (i - lead) as u64));
            }
            let clock = Instant::now();
            std::hint::black_box(decoder.push_features(&model, std::hint::black_box(&rows[i])));
            spent += clock.elapsed();
        }
        kept.push(ms(spent));
    }

    let mut redone = Vec::with_capacity(total);
    let mut bytes = vec![[0i8; FEATURES]; rows.len()];
    for (row, out) in rows.iter().zip(bytes.iter_mut()).take(lead) {
        model.scale_row(row, out);
    }
    let start = Instant::now();
    for step in 0..total {
        let newest = lead + (step + 1) * HOP; // one past the newest row of this step
        if paced {
            wait_until(start + Duration::from_millis(ROW_MS * ((step + 1) * HOP - 1) as u64));
        }
        let clock = Instant::now();
        for i in newest - HOP..newest {
            model.scale_row(std::hint::black_box(&rows[i]), &mut bytes[i]);
        }
        let window: &[[i8; FEATURES]; WINDOW] = bytes[newest - WINDOW..newest].try_into().unwrap();
        std::hint::black_box(decoder.step_from_scratch(&model, window));
        redone.push(ms(clock.elapsed()));
    }

    let list = |times: &[f64]| times[warmup..].iter().map(|t| format!("{t:.6}")).collect::<Vec<_>>().join(",");
    println!(
        "{{\"width\":{C},\"decoder_bytes\":{},\"model_bytes\":{},\"paced\":{paced},\"kept_ms\":[{}],\"redone_ms\":[{}]}}",
        core::mem::size_of::<Decoder<C>>(),
        file.len(),
        list(&kept),
        list(&redone)
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: host check <model.bin>  |  host time <model.bin> <steps> <warmup> <paced: 0 or 1>";
    if args.len() < 3 {
        panic!("{usage}");
    }
    let file = std::fs::read(&args[2])?;
    assert!(file.len() >= 12, "{} is too short to be a model file", args[2]);
    let width = u32::from_le_bytes(file[8..12].try_into().unwrap());
    // The decoder's width is fixed when it is compiled, so each width is its own copy of the code.
    macro_rules! for_width {
        ($($w:literal),*) => {
            match (args[1].as_str(), width) {
                $(("check", $w) => check::<$w>(&file),
                  ("time", $w) if args.len() == 6 => time::<$w>(
                      &file, args[3].parse().expect("steps"), args[4].parse().expect("warmup"), args[5] == "1"),)*
                _ => panic!("{usage}\n(model width {width}; this build knows widths 256, 128, 64, 32 and 16)"),
            }
        };
    }
    for_width!(256, 128, 64, 32, 16)
}

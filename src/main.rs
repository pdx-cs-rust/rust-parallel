use std::ops::Range;

use image::{ImageBuffer, Luma, codecs::png::PngEncoder};
use ndarray::{Array2, Axis};
use num::Complex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use clap::Parser;


#[derive(Debug, Clone)]
struct Dimensions {
    width: usize,
    height: usize,
}

// Implementation by DeepSeek.
impl std::str::FromStr for Dimensions {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('x').collect();
        if parts.len() != 2 {
            eprintln!("invalid dimensions format: expected <width>x<height>.");
            std::process::exit(1);
        }

        let width = parts[0].parse::<usize>()?;
        let height = parts[1].parse::<usize>()?;

        Ok(Dimensions { width, height })
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value="80x20")]
    dims: Dimensions,
    #[arg(short, long, default_value="255")]
    bound: u16,
    #[arg(short, long)]
    ascii: bool,
    filename: Option<std::path::PathBuf>,
}

#[cfg(feature = "rayon")]
macro_rules! gen_iter_mut {
    ($r:expr) => { $r.par_bridge() };
}

#[cfg(not(feature = "rayon"))]
macro_rules! gen_iter_mut {
    ($r:expr) => { $r };
}

type C = Complex<f64>;

fn escapes(bound: u16, c: C) -> Option<u16> {
    let mut z = Complex::new(0.0, 0.0);
    for g in 0..bound {
        if z.norm_sqr() > 4.0 {
            return Some(g)
        }
        z = z * z + c;
    }
    None
}

#[test]
fn test_escapes() {
    let zero = C::new(0.0, 0.0);
    assert!(matches!(escapes(10, zero), None));
    let two = C::new(2.0, 2.0);
    assert!(matches!(escapes(10, two), Some(_)));
}

type R = Range<f64>;

fn mandelbrot(
    bound: u16,
    (width, height): (usize, usize),
    (xz, yz): (R, R),
) -> Array2<u16> {
    let mut result = Array2::zeros((height, width));
    let x_step = (xz.end - xz.start) / width as f64;
    let y_step = (yz.end - yz.start) / width as f64;
    let rows = gen_iter_mut!(result.axis_iter_mut(Axis(0)).enumerate());
    rows.for_each(move |(j, mut row)| {
        let y = yz.start + j as f64 * y_step;
        for (i, v) in row.iter_mut().enumerate() {
            let x = xz.start + i as f64 * x_step;
            let c = Complex::new(x, y);
            let e = escapes(bound, c).unwrap_or(bound + 1);
            let es = e as f64 * u16::MAX as f64 / (bound + 1) as f64;
            *v = es.floor() as u16;
        }
    });
    result
}

fn open_file(filename: Option<std::path::PathBuf>) -> Box<dyn std::io::Write> {
    match filename {
        None => Box::new(std::io::stdout()),
        Some(p) => match std::fs::File::create(p) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("output file: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn display(f: &mut dyn std::io::Write, a: Array2<u16>) {
    for row in a.rows() {
        for &v in row {
            let c = if v < 6554 {
                ' '
            } else if v < 32767 {
                '.'
            } else {
                '*'
            };
            write!(f, "{}", c).unwrap();
            
        }
        writeln!(f).unwrap();
    }
}

fn render(f: &mut dyn std::io::Write, a: Array2<u16>) {
    let width = a.ncols() as u32;
    let height = a.nrows() as u32;
    let (pixels, offset) = a.into_raw_vec_and_offset();
    assert!(matches!(offset, Some(0)));
    let img: ImageBuffer<Luma<u16>, _> = ImageBuffer::from_raw(width, height, pixels).unwrap();
    let encoder = PngEncoder::new(f);
    img.write_with_encoder(encoder).unwrap();
}

fn sum(a: Array2<u16>) -> u16 {
    a.into_iter().fold(0, |acc, v| acc.wrapping_add(v))
}

fn main() {
    let args = Args::parse();
    let width = args.dims.width;
    let height = args.dims.height;

    let ratio = width as f64 / height as f64;
    let r = (
        (-1.0 * ratio..1.0 * ratio),
        (-1.0..1.0),
    );

    let m = mandelbrot(args.bound, (width, height), r);
    if args.ascii {
        display(&mut open_file(args.filename), m);
    } else {
        if args.filename.is_some() {
            render(&mut open_file(args.filename), m);
        } else {
            println!("{}", sum(m));
        }
    }
}

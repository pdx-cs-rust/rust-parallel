use std::ops::Range;

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
            *v = escapes(bound, Complex::new(x, y)).unwrap_or(bound + 1);
        }
    });
    result
}

fn display(a: &Array2<u16>) {
    for row in a.rows() {
        for v in row {
            let c = match v {
                0..10 => ' ',
                10..100 => '.',
                _ => '*',
            };
            print!("{}", c);
            
        }
        println!();
    }
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
    display(&m);
}

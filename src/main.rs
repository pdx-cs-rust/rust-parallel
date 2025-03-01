use std::ops::Range;

use ndarray::{Array2, Axis};
use num::Complex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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
    let mut result = Array2::zeros((width, height));
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
    let width = 8000;
    let height = 1000;
    let ratio = width as f64 / height as f64;
    let r = (
        (-1.0 * ratio..1.0 * ratio),
        (-1.0..1.0),
    );
    let m = mandelbrot(255, (width, height), r);
    display(&m);
}

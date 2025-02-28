use std::ops::Range;

use num::{Complex};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

const WIDTH: usize = 8000;
const HEIGHT: usize = 1000;

#[cfg(feature = "rayon")]
macro_rules! gen_iter_mut {
    ($r:expr) => { $r.par_iter_mut() };
}
#[cfg(not(feature = "rayon"))]
macro_rules! gen_iter_mut {
    ($r:expr) => { $r.iter_mut() };
}

type C = Complex<f64>;

fn escapes(bound: usize, c: C) -> Option<usize> {
    let mut z = Complex::new(0.0, 0.0);
    for g in 0..=bound {
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
type Image = [[usize; WIDTH]; HEIGHT];


fn mandelbrot(bound: usize, (xz, yz): (R, R)) -> Box<Image>
{
    let mut result: Box<Image> = Box::new([[0; WIDTH]; HEIGHT]);
    let x_step = (xz.end - xz.start) / WIDTH as f64;
    let y_step = (yz.end - yz.start) / HEIGHT as f64;
    // Thanks to DeepSeek for help with the rayon.
    gen_iter_mut!(result).enumerate().for_each(move |(j, row)| {
        let y = yz.start + j as f64 * y_step;
        #[allow(clippy::needless_range_loop)]
        gen_iter_mut!(row).enumerate().for_each(move |(i, v)| {
            let x = xz.start + i as f64 * x_step;
            *v = escapes(bound, Complex::new(x, y)).unwrap_or(bound + 1);
        });
    });
    result
}

fn display(a: &Image) {
    for row in a {
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
    let ratio = WIDTH as f64 / HEIGHT as f64;
    let r = (
        (-1.0 * ratio..1.0 * ratio),
        (-1.0..1.0),
    );
    let m = mandelbrot(255, r);
    display(&m);
}

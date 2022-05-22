//! # What is this?
//! This is a Rust library for linear algebra, it is completely generic and can be used for any type of numeric matrix
//! # Why is this in Rust?
//! This is because Rust is a statically typed language, and it is a good way to learn how to use Rust, it also allows
//! for static sizing of matricies at compile time, this is a hughe advantage over dynamic sizing in other languages
//! # How do I use this?
//! This library will be used for the neural network library, but it can be used for other things as well.
//! Ideally I would want to use this for embedded systems but due to the lack of fpu's It would need special care,
//! this could be done by not casting the matrix to a f32 matrix, but instead using the u32 values or similar instead,
//! this would however require a lot of rethinking since neural nets usually work with values in the range of 0-1,
//! and this would require a lot of rethinking of the math behind the neural nets.
//! # How do I use this in Rust?
//! ```rust
//! use matrs::matrix::Matrix;
//! let m1 = Matrix::<f32, 2, 2>::new();
//! let m2 = Matrix::<f32, 2, 2>::new();
//! let m3 = m1 + m2;
//! ```
//! # Future work
//! This library is currently in a very early stage of development, I would like to add more features to it,
//! I would love to add support for interopperability with c and python.
//! I would love to add support for gpu / tpu / other hardware accelerators.
//!
//! # License
//! This library is licensed under the MIT license, see the LICENSE file for more information.
//! It provides absolutely no warranty, and is provided as is.
//! # Contributing
//! This project is open source, feel free to contribute!
//!
//! # Credits
//! This library was written by [@ivario123](Ivar Jönsson)
pub mod matrix;
pub mod traits;
pub mod vec;

// Defining some initer operation of matrix and vector objects

use crate::matrix::*;
use crate::traits::CompliantNumerical;
use crate::vec::*;
use std::ops::Mul;
// Defines a method to transform a vector with a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Mul<Matrix<T, ROWS, COLS>>
    for Vec<T, ROWS>
{
    type Output = Vec<T, ROWS>;
    fn mul(self, other: Matrix<T, ROWS, COLS>) -> Vec<T, ROWS> {
        let mut result = Vec::new();
        for row in 0..ROWS {
            let mut sum: T = T::default();
            for col in 0..COLS {
                sum = sum + *self.get(row) * *other.get(row, col);
            }
            result.set(row, sum);
        }
        result
    }
}

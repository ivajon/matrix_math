#![no_std]
pub mod matrix;
pub mod traits;
pub mod vec;

#[cfg(feature = "experimental")]
#[doc(hidden)]
pub mod vector_matrix;
#[cfg(feature = "experimental")]
#[doc(hidden)]
pub mod x86_optimization;

pub mod predule {
    pub use super::matrix::Matrix;
    pub use super::traits::{MatrixInterface, VectorItterator, VectorTrait};
    pub use super::vec::Vector;
}

// Defining some initer operation of matrix and vector objects

use crate::matrix::*;
pub use crate::traits::{CompliantNumerical, MatrixInterface, VectorTrait};
use crate::vec::*;
use core::ops::Mul;
// Defines a method to transform a vector with a matrix
impl<
        T: CompliantNumerical + Mul<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > Mul<Matrix<TOther, ROWS, COLS>> for Vector<T, ROWS>
{
    type Output = Vector<T, ROWS>;
    /// Defines a method to transform a vector with a matrix
    fn mul(self, other: Matrix<TOther, ROWS, COLS>) -> Vector<T, ROWS> {
        let mut result = Vector::new();
        for row in 0..ROWS {
            let mut sum: T = T::default();
            for col in 0..COLS {
                sum += self.get(row).clone() * other.get(row, col).clone();
            }
            result.set(row, sum);
        }
        result
    }
}

impl<
        T: CompliantNumerical + Mul<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > Mul<Vector<TOther, COLS>> for Matrix<T, ROWS, COLS>
{
    type Output = Vector<T, ROWS>;
    /// Defines a method to multiply a matrix with a vector
    fn mul(self, other: Vector<TOther, COLS>) -> Vector<T, ROWS> {
        let mut result = Vector::new();
        for row in 0..ROWS {
            let mut sum: T = T::default();
            for col in 0..COLS {
                sum += self.get(row, col).clone() * other.get(col).clone();
            }
            result.set(row, sum);
        }
        result
    }
}

impl<T: CompliantNumerical, const COUNT: usize> Vector<T, COUNT> {
    /// Creates a matrix from a vector

    pub fn to_matrix(self) -> Matrix<T, COUNT, 1> {
        let mut result = Matrix::new();
        for i in 0..COUNT {
            result.set(i, 0, self.get(i).clone());
        }
        result
    }
}

impl<T: CompliantNumerical, const ROWS: usize> Matrix<T, ROWS, 1> {
    /// Creates a vector from a matrix
    pub fn row_vec(self) -> Vector<T, ROWS> {
        let mut result = Vector::new();
        for i in 0..ROWS {
            result.set(i, self.get(i, 0).clone());
        }
        result
    }
}

impl<T: CompliantNumerical, const COLS: usize> Matrix<T, 1, COLS> {
    /// Creates a vector from a matrix
    pub fn col_vec(self) -> Vector<T, COLS> {
        let mut result = Vector::new();
        for i in 0..COLS {
            result.set(i, self.get(0, i).clone());
        }
        result
    }
}
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Creates an array of vectors from a matrix

    pub fn to_array_of_vecs(self) -> [Vector<T, COLS>; ROWS] {
        let mut result: [Vector<T, COLS>; ROWS] = array_init::array_init(|_| Vector::new());
        let data = self.get_elements();
        for i in 0..ROWS {
            result[i] = Vector::new_from_data(data[i].clone());
        }
        result
    }
}

pub mod matrix;
pub mod traits;
pub mod vec;
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

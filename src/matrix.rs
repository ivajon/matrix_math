///! Defines a matrix type
///! It is a fixed size matrix, with a user defined amount of rows and columns
///! It allows for anny type that supports the VecElement trait
///! It is used to implement linear algebra for neural nets and similar
use std::{
    ops::{self},
    usize,
};

use crate::traits::CompliantNumerical;
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T: CompliantNumerical, const ROWS: usize, const COLS: usize> {
    elements: [[T; COLS]; ROWS],
}
#[allow(dead_code)]
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn new() -> Matrix<T, ROWS, COLS> {
        let elements = [[T::default(); COLS]; ROWS];
        Matrix { elements }
    }
    pub fn new_from_data(data: [[T; COLS]; ROWS]) -> Matrix<T, ROWS, COLS> {
        Matrix { elements: data }
    }
    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.elements[row][col]
    }
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.elements[row][col]
    }
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.elements[row][col] = value;
    }
    pub fn transpose(&mut self) -> Matrix<T, COLS, ROWS> {
        let mut m = Matrix::<T, COLS, ROWS>::new();
        // Square matrix, so we can just swap the rows and columns
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(col, row, self.elements[row][col].clone());
            }
        }
        m
    }
}
#[allow(dead_code)]
/// Defines a add method for the generic matrix struct
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Add<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    fn add(self, other: Matrix<T, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                result.set(row, col, *self.get(row, col) + *other.get(row, col));
            }
        }
        result
    }
}
#[allow(dead_code)]
/// Defines a sub method for the generic matrix struct
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Sub<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    fn sub(self, other: Matrix<T, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                result.set(row, col, *self.get(row, col) - *other.get(row, col));
            }
        }
        result
    }
}
#[allow(dead_code)]
/// Defines a mul method for the generic matrix struct
/// This is the implementation for the multiplication of a matrix with a scalar
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Mul<T>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    fn mul(self, other: T) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                result.set(row, col, *self.get(row, col) * other);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix_new() {
        let m = Matrix::<u32, 1, 1>::new();
        assert_eq!(m.get(0, 0), &0);
    }
    #[test]
    fn test_matrix_new_from_data() {
        let data = [[1, 2, 3]; 3];
        let m = Matrix::new_from_data(data);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(0, 1), &2);
        assert_eq!(m.get(0, 2), &3);
        assert_eq!(m.get(1, 0), &1);
        assert_eq!(m.get(1, 1), &2);
        assert_eq!(m.get(1, 2), &3);
        assert_eq!(m.get(2, 0), &1);
        assert_eq!(m.get(2, 1), &2);
        assert_eq!(m.get(2, 2), &3);
    }
    #[test]
    fn test_matrix_set() {
        let mut m = Matrix::<u32, 1, 1>::new();
        m.set(0, 0, 1);
        assert_eq!(m.get(0, 0), &1);
    }
    #[test]
    fn test_matrix_add() {
        let mut m1 = Matrix::<u32, 1, 1>::new();
        let mut m2 = Matrix::<u32, 1, 1>::new();
        m1.set(0, 0, 1);
        m2.set(0, 0, 1);
        let m3 = m1 + m2;
        assert_eq!(m3.get(0, 0), &2);
    }
    #[test]
    fn test_matrix_sub() {
        let mut m1 = Matrix::<i32, 1, 1>::new();
        let mut m2 = Matrix::<i32, 1, 1>::new();
        m1.set(0, 0, 0);
        m2.set(0, 0, 1);
        let m3 = m1 - m2;
        assert_eq!(m3.get(0, 0), &-1);
    }
    #[test]
    fn test_matrix_mul_scalar() {
        let mut m1 = Matrix::<u32, 1, 1>::new();
        m1.set(0, 0, 1);
        let m2 = m1 * 2;
        assert_eq!(m2.get(0, 0), &2);
    }
    #[test]
    fn test_matrix_transpose() {
        let data = [[1, 2, 3]; 3];
        let mut m = Matrix::new_from_data(data);
        m = m.transpose();

        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &2);
        assert_eq!(m.get(2, 0), &3);
        assert_eq!(m.get(0, 1), &1);
        assert_eq!(m.get(1, 1), &2);
        assert_eq!(m.get(2, 1), &3);
        assert_eq!(m.get(0, 2), &1);
        assert_eq!(m.get(1, 2), &2);
        assert_eq!(m.get(2, 2), &3);
    }
}

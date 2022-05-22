//! This files defines a generic matrix struct, it is used to represent a matrix of any numeric type,
//! it is generic in the type of the matrix, and it is generic in the number of rows and columns
//! of the matrix.
//!
//! It also allocates memory statically at compile time, this is a huge advantage over dynamic
//! memory allocation.
//!
//! # Why should I use this?
//! This library is specifically designed to work on low memory devices, such as embedded systems,
//! but since it is designed that way, it should also perform well on high memory devices.
//!
//! # How do I use this?
//! ```rust
//! use matrs::matrix::Matrix;
//! let m1 = Matrix::<f32, 2, 2>::new();
//! let m2 = Matrix::<f32, 2, 2>::new();
//! let m3 = m1 + m2;
//! ```
use std::{
    ops::{self},
    usize,
};

use crate::traits::{self, CompliantNumerical};
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> {
    elements: [[T; COLS]; ROWS],
}
#[allow(dead_code)]
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Creates a new matrix of the specified size
    pub fn new() -> Matrix<T, ROWS, COLS> {
        let elements = [[T::default(); COLS]; ROWS];
        Matrix { elements }
    }
    /// Instantiantes a new matrix with the given elements
    pub fn new_from_data(data: [[T; COLS]; ROWS]) -> Matrix<T, ROWS, COLS> {
        Matrix { elements: data }
    }
    /// Get the value of the element at the given row and column
    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.elements[row][col]
    }
    /// Returns a mutable reference to the element at the given row and column
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.elements[row][col]
    }
    /// Sets the value of a given element in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.elements[row][col] = value;
    }
    /// Transposes a given matrix, since we can't assume the matrix to be square, this function takes O(n) memory
    /// and takes O(n^2) time to transpose a matrix, if the matrix is square this could be done in O(1) memory and theta(n^2 / 2) time, which is similar but better
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

    /// Defines a gradient method for the generic matrix struct
    pub fn gradient(&self, f: impl Fn(T) -> T) -> Matrix<T, ROWS, COLS> {
        let mut m = Matrix::<T, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, f(self.elements[row][col].clone()));
            }
        }
        m
    }
    /// Converts an integer matrix to a float32 matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<i32, 2, 2>::new();
    /// let mf = m.to_f32();
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///        [0.0, 0.0]])
    /// ```

    pub fn to_f32(&self) -> Matrix<f32, ROWS, COLS> {
        let mut m = Matrix::<f32, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, self.elements[row][col].into_f32());
            }
        }
        m
    }
    /// Converts an integer matrix to a float64 matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<i32, 2, 2>::new();
    /// let mf = m.to_f64();
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///       [0.0, 0.0]])
    /// ```

    pub fn to_f64(&self) -> Matrix<f64, ROWS, COLS> {
        let mut m = Matrix::<f64, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, self.elements[row][col].into_f64());
            }
        }
        m
    }

    // Converting a float matrix to an integer matrix is kinda pointless for reliable systems since it looses precision,
    // It also carries some risk for panics if not implemented correctly, so it is not implemented
}
#[allow(dead_code)]
/// Defines a add method for the generic matrix struct
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::Add<Matrix<T, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds two matrices together
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m1 = Matrix::<f32, 2, 2>::new();
    /// let m2 = Matrix::<f32, 2, 2>::new();
    /// let m3 = m1 + m2;
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///       [0.0, 0.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    /// # Performance
    /// This method is O(n^2) time and O(1) memory
    /// # Remarks
    /// This method gaurantees at compile time that the matrices are the same size
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
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::Sub<Matrix<T, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method subtracts two matrices
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m1 = Matrix::<f32, 2, 2>::new();
    /// let m2 = Matrix::<f32, 2, 2>::new();
    /// let m3 = m1 - m2;
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///      [0.0, 0.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    /// # Performance
    /// This method is O(n^2) time and O(1) memory
    /// # Remarks
    /// This method gaurantees at compile time that the matrices are the same size

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
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Mul<T>
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
// Multiplying an integer matrix with a float matrix is nonsensical, so it is not implemented
// If you want to do that first convert the integer matrix to a float matrix
impl<T: CompliantNumerical, const ROWS: usize, const OWN_COLS: usize, const OTHERS_COLS: usize>
    std::ops::Mul<Matrix<T, ROWS, OTHERS_COLS>> for Matrix<T, OWN_COLS, ROWS>
{
    type Output = Matrix<T, OWN_COLS, OTHERS_COLS>;
    #[inline(never)]
    /// # What is this?
    /// This is the implementation for the multiplication of a matrix with another matrix
    /// # Why should I use this?
    /// This implementation gaurantees that the dimensions of the matrices are correct, so that the
    /// multiplication can be done without any errors. This falls directly from the fact
    /// that the matricies are statically sized, so the compiler can check the dimensions
    fn mul(self, other: Matrix<T, ROWS, OTHERS_COLS>) -> Matrix<T, OWN_COLS, OTHERS_COLS> {
        let mut result = Matrix::new();
        for row in 0..OWN_COLS {
            for col in 0..OTHERS_COLS {
                let mut sum = T::default();
                for i in 0..ROWS {
                    sum = sum + *self.get(row, i) * *other.get(i, col);
                }
                result.set(row, col, sum);
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
    #[test]
    fn test_matrix_mul_matrix() {
        let data1 = [[1, 2, 3]; 3];
        let m1 = Matrix::new_from_data(data1);
        let data2 = [[1, 2, 3]; 3];
        let m2 = Matrix::new_from_data(data2);
        let m3 = m1 * m2;
        println!("{:?}", m3);
        assert_eq!(m3.get(0, 0), &6);
        assert_eq!(m3.get(0, 1), &12);
        assert_eq!(m3.get(0, 2), &18);
        assert_eq!(m3.get(1, 0), &6);
        assert_eq!(m3.get(1, 1), &12);
        assert_eq!(m3.get(1, 2), &18);
        assert_eq!(m3.get(2, 0), &6);
        assert_eq!(m3.get(2, 1), &12);
        assert_eq!(m3.get(2, 2), &18);
    }
    #[test]
    fn test_matrix_mul_matrix_non_square() {
        let data1: [[u32; 3]; 4] = [[1, 2, 3]; 4];
        let m1 = Matrix::new_from_data(data1).to_f32();
        let data2: [[i32; 4]; 3] = [[1, 2, 3, 4]; 3];
        let m2 = Matrix::new_from_data(data2).to_f32();
        let m3 = m1 * m2;
        assert_eq!(m3.get(0, 0), &6.0);
        assert_eq!(m3.get(0, 1), &12.0);
        assert_eq!(m3.get(0, 2), &18.0);
        assert_eq!(m3.get(0, 3), &24.0);
        assert_eq!(m3.get(1, 0), &6.0);
        assert_eq!(m3.get(1, 1), &12.0);
        assert_eq!(m3.get(1, 2), &18.0);
        assert_eq!(m3.get(1, 3), &24.0);
        assert_eq!(m3.get(2, 0), &6.0);
        assert_eq!(m3.get(2, 1), &12.0);
        assert_eq!(m3.get(2, 2), &18.0);
        assert_eq!(m3.get(2, 3), &24.0);
    }
}

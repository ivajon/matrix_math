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
    ops::{self, Index, IndexMut},
    usize,
};

use crate::traits::{self, CompliantNumerical};
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> {
    elements: [[T; COLS]; ROWS],
}
pub type Matrix2x2<T> = Matrix<T, 2, 2>;
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
pub type Matrix4x4<T> = Matrix<T, 4, 4>;

#[allow(dead_code)]
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Creates a new matrix of the specified size
    pub fn new() -> Matrix<T, ROWS, COLS> {
        assert_ne!(ROWS, 0, "Rows must be greater than 0");
        assert_ne!(COLS, 0, "Columns must be greater than 0");
        let elements = [[T::default(); COLS]; ROWS];
        Matrix { elements }
    }
    /// Creates a new matrix of zeros of the specified size
    pub fn zeros() -> Matrix<T, ROWS, COLS> {
        Matrix::new()
    }
    /// Instantiantes a new matrix with the given elements
    pub fn new_from_data(data: [[T; COLS]; ROWS]) -> Matrix<T, ROWS, COLS> {
        assert_ne!(ROWS, 0, "ROWS must be greater than 0");
        assert_ne!(COLS, 0, "COLS must be greater than 0");
        Matrix { elements: data }
    }
    /// Get the value of the element at the given row and column
    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.elements[row][col]
    }
    /// Get the mutable value of the element at the given row and column
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m = Matrix::<f32, 2, 2>::new();
    /// let mut val = m.get_mut(0, 0);
    /// *val = 1.0;
    /// ```
    /// # Mathematical equivalent
    /// ```latex
    /// A_{ij} = 1
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.elements[row][col]
    }
    /// Returns the entire elements array
    /// This is useful for when you want to do something with the entire matrix
    /// without having to use the get and set functions
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 2>::new();
    /// let elements = m.get_elements();
    /// ```
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m = Matrix::<f32, 2, 2>::new();
    /// m.set_elements([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut elements = m.get_elements();
    /// assert_eq!(*elements, [[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    pub fn get_elements(&self) -> &[[T; COLS]; ROWS] {
        &self.elements
    }
    /// Returns the number of rows in the matrix
    /// # Note
    /// This is a constant, so it can be used in a const context
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 2>::new();
    /// assert_eq!(m.rows(), 2);
    /// ```
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 3>::new();
    /// assert_eq!(m.rows(), 2);
    /// ```
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 3, 2>::new();
    /// assert_eq!(m.rows(), 3);
    /// ```
    #[inline(always)]
    pub fn rows(&self) -> usize {
        ROWS
    }
    /// Returns the number of columns in the matrix
    /// # Note
    /// This should be const, but since we can't use const generics, we have to use usize
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 2>::new();
    /// assert_eq!(m.cols(), 2);
    /// ```
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 3>::new();
    /// assert_eq!(m.cols(), 3);
    /// ```
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 3, 2>::new();
    /// assert_eq!(m.cols(), 2);
    /// ```
    #[inline(always)]
    pub fn cols(&self) -> usize {
        COLS
    }
    /// Sets the value of a given element in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.elements[row][col] = value;
    }

    /// Sets the entire elements array
    /// This is useful for when you want to do something with the entire matrix
    /// without having to use the get and set functions
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m = Matrix::<f32, 2, 2>::new();
    /// m.set_elements([[1.0, 2.0], [3.0, 4.0]]);
    /// ```

    pub fn set_elements(&mut self, data: [[T; COLS]; ROWS]) {
        self.elements = data.clone();
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
    /// Passes every element of the matrix to a function defined as
    /// ```rust
    /// fn f(x: f64) -> f64 {
    ///     x.exp()
    /// }
    /// ```

    pub fn map<F>(&self, f: F) -> Matrix<T, ROWS, COLS>
    where
        F: Fn(T) -> T,
    {
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
    /// assert_eq!(*mf.get_elements(), [[0.0, 0.0], [0.0, 0.0]]);
    ///
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///       [0.0, 0.0]])
    /// ```
    /// # Safety
    /// This function is safe since it is only used to convert integer matrices to float64 matrices
    /// # Panics
    /// This function will never panic
    /// # Examples
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<i32, 2, 2>::new();
    /// let mf = m.to_f64();
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///      [0.0, 0.0]])
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

    /// Converts a float32 matrix to an integer matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 2>::new();
    /// let mi = m.to_i32();
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0, 0],
    ///      [0, 0]])
    /// ```

    pub fn to_i32(&self) -> Matrix<i32, ROWS, COLS> {
        let mut m = Matrix::<i32, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, self.elements[row][col].into_i32());
            }
        }
        m
    }
    /// Converts a float64 matrix to an integer matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m = Matrix::<f64, 2, 2>::new();
    /// m[(0,0)] = 1.0;
    /// let mi = m.to_i64();
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[1, 0],
    ///     [0, 0]])
    /// ```

    pub fn to_i64(&self) -> Matrix<i64, ROWS, COLS> {
        let mut m = Matrix::<i64, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, self.elements[row][col].into_i64());
            }
        }
        m
    }

}
impl<T: CompliantNumerical, const DIMENSION: usize> Matrix<T, DIMENSION, DIMENSION> {
    /// Returns the identity matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f64, 2, 2>::eye();
    /// assert_eq!(*m.get_elements(), [[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    pub fn eye() -> Matrix<T, DIMENSION, DIMENSION> {
        assert_ne!(DIMENSION, 0);
        let mut m = Matrix::<T, DIMENSION, DIMENSION>::new();
        for row in 0..DIMENSION {
            m.set(row, row, T::from_i64(1));
        }
        m
    }
    /// Just an alias for the eye function
    pub fn identity() -> Matrix<T, DIMENSION, DIMENSION> {
        Matrix::eye()
    }
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
// Implementing the add method for the generic matrix struct and a scalar
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Add<T>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds a scalar to a matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m1 = Matrix::<f32, 2, 2>::new();
    /// let m2 = m1 + 1.0;
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[1.0, 1.0],
    ///       [1.0, 1.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    /// # Performance
    /// This method is O(n^2) time and O(1) memory
    /// # Remarks
    /// This method gaurantees at compile time that the matrices are the same size
    fn add(self, other: T) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                result.set(row, col, *self.get(row, col) + other);
            }
        }
        result
    }
}

// Allows for add assign to be used on a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::AddAssign<Matrix<T, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a matrix to a matrix and stores the result in the first matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// let m2 = Matrix::<f32, 2, 2>::new();
    /// m1 += m2;
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
    fn add_assign(&mut self, other: Matrix<T, ROWS, COLS>) {
        for row in 0..ROWS {
            for col in 0..COLS {
                self.set(row, col, *self.get(row, col) + *other.get(row, col));
            }
        }
    }
}

// Allows for add assign to be used on a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::AddAssign<T>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a scalar to a matrix and stores the result in the first matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// m1 += 1.0;
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[1.0, 1.0],
    ///       [1.0, 1.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    /// # Performance
    /// This method is O(n^2) time and O(1) memory
    /// # Remarks
    /// This method gaurantees at compile time that the matrices are the same size
    fn add_assign(&mut self, other: T) {
        self.elements = (*self + other).get_elements().clone();
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
// Implements sub assign for a matrix and a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::SubAssign<Matrix<T, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method subtracts two matrices
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// let m2 = Matrix::<f32, 2, 2>::new();
    /// m1 -= m2;
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
    fn sub_assign(&mut self, other: Matrix<T, ROWS, COLS>) {
        self.elements = (*self - other).elements;
    }
}
// Implements multiplication for a matrix and a scalar
#[allow(dead_code)]
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Mul<T>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method multiplies a matrix with a scalar
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// m1[(1,1)] = 2.0;
    /// let m2 = m1 * 2.0;
    /// assert_eq!(m2[(1,1)], 4.0);
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///     [0.0, 2.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
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
// Implements mul assign
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> std::ops::MulAssign<T>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method multiplies a matrix with a scalar
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// m1[(1,1)] = 2.0;
    /// m1 *= 2.0;
    /// assert_eq!(m1[(1,1)], 4.0);
    /// ```
    /// # Output
    /// ```text
    /// Matrix([[0.0, 0.0],
    ///     [0.0, 2.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    fn mul_assign(&mut self, other: T) {
        self.elements = (*self * other).elements;
    }
}

// Multiplying an integer matrix with a float matrix is nonsensical, so it is not implemented
// If you want to do that first convert the integer matrix to a float matrix or vice versa
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
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m1 = Matrix::<f32, 2, 2>::new();
    /// let mut m2 = Matrix::<f32, 2, 2>::new();
    /// m1[(0,0)] = 1.0;
    /// m1[(1,1)] = 1.0;
    /// m2[(0,0)] = 2.0;
    /// m2[(0,1)] = 2.0;
    /// m2[(1,0)] = 2.0;
    /// m2[(1,1)] = 2.0;
    ///
    /// let m3 = m1 * m2;
    /// assert_eq!(m3.get_elements(),m2.get_elements());
    /// ```
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
// Implements indexing for the matrix
impl<T: traits::CompliantNumerical, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for Matrix<T, ROWS, COLS>
{
    type Output = T;
    /// Allows for array like access to the matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let m = Matrix::<f32, 2, 2>::new();
    /// let point = m[(0, 1)];
    /// ```
    /// # Output
    /// ```text
    /// 0.0
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    /// # Performance
    /// This method is O(1) time and O(1) memory
    fn index(&self, index: (usize, usize)) -> &T {
        &self.get(index.0, index.1)
    }
}
// Implements index mut
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)>
    for Matrix<T, ROWS, COLS>
{
    /// Allows for array like assignment to the matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::Matrix;
    /// let mut m = Matrix::<f32, 2, 2>::new();
    /// m[(0, 1)] = 1.0;
    /// ```
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        self.get_mut(index.0, index.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::*;
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

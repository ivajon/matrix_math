//! This files defines a generic matrix struct, it is used to represent a matrix of any numeric type,
//! it is generic in the type of the matrix, and it is generic in the number of rows and columns
//! of the matrix.
//!
//! It also allocates memory statically at compile time, this is a huge advantage over dynamic
//! memory allocation.
//!
//!
//! # Why should I use this?
//! This library is specifically designed to work on low memory devices, such as embedded systems,
//! but since it is designed that way, it should also perform well on high memory devices.
//! # Notes
//! This generic uses unwrap, this should not be done since we can miss errors.
//! # TODO
//! Remove the unwrap

use crate::traits::{CompliantNumerical, MatrixInterface};
use core::{ops::*, usize};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
/// # Matrix
/// This is a generic matrix struct, it is used to represent a matrix of any numeric type,
/// it is generic in the type of the matrix, and it is generic in the number of rows and columns
/// of the matrix.
///
///
pub struct Matrix<T: CompliantNumerical, const ROWS: usize, const COLS: usize> {
    /// This is the internal data of the matrix, it is a fixed size array of the type T
    /// that is ROWS * COLS in size.
    /// It is a fixed size array, so that it can be allocated at compile time.
    ///
    /// This field is not exposed to the user, and is only used internally, if exposed to the
    /// user it's through the getter and setter functions.
    elements: [[T; COLS]; ROWS],
}
/// Simpeler alias for a 2x2 matrix
pub type Matrix2x2<T> = Matrix<T, 2, 2>;
/// Simpeler alias for a 3x3 matrix
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
/// Simpeler alias for a 4x4 matrix
pub type Matrix4x4<T> = Matrix<T, 4, 4>;

#[allow(dead_code)]
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> MatrixInterface<T, ROWS, COLS>
    for Matrix<T, ROWS, COLS>
{
    /// Creates a new matrix of the specified size
    fn new() -> Matrix<T, ROWS, COLS> {
        let elements = [[T::default(); COLS]; ROWS];
        Matrix { elements }
    }
    /// Instantiantes a new matrix with the given elements
    fn new_from_data(data: [[T; COLS]; ROWS]) -> Matrix<T, ROWS, COLS> {
        Matrix { elements: data }
    }
    // ================================================================
    // Getters
    // ================================================================
    /// Get the value of the element at the given row and column
    fn get(&self, row: usize, col: usize) -> &T {
        &self.elements[row][col]
    }
    /// Get the mutable value of the element at the given row and column
    #[inline(always)]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.elements[row][col]
    }
    // ================================================================
    // Setters
    // ================================================================
    /// Sets the value of a given element in the matrix
    fn set(&mut self, row: usize, col: usize, value: T) {
        self.elements[row][col] = value;
    }

    /// Sets the entire elements array
    /// This is useful for when you want to do something with the entire matrix
    /// without having to use the get and set functions

    fn set_elements(&mut self, data: [[T; COLS]; ROWS]) {
        self.elements = data.clone();
    }
    // ================================================================
    // Convience functions
    // ================================================================
    /// Transposes a given matrix, since we can't assume the matrix to be square, this function takes O(n) memory
    /// and takes O(n^2) time to transpose a matrix, if the matrix is square this could be done in O(1) memory and theta(n^2 / 2) time, which is similar but better
    fn transpose(&mut self) -> Matrix<T, COLS, ROWS> {
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

    fn map<F: Fn(T) -> T>(&self, f: F) -> Self {
        let mut m = Matrix::<T, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                m.set(row, col, f(self[(row, col)].clone()));
            }
        }
        m
    }

    fn identity() -> Self {
        let mut m = Matrix::<T, ROWS, COLS>::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                if row == col {
                    m.set(row, col, T::one());
                } else {
                    m.set(row, col, T::zero());
                }
            }
        }
        m
    }

    type TransposeOutput = Matrix<T, COLS, ROWS>;
}

impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn get_elements(&self) -> [[T; COLS]; ROWS] {
        self.elements.clone()
    }
}
// ================================================================
// Matrix Operations implementations
// ================================================================
impl<T: CompliantNumerical, const DIMENSION: usize> Matrix<T, DIMENSION, DIMENSION> {
    /// Returns the identity matrix
    pub fn eye() -> Matrix<T, DIMENSION, DIMENSION> {
        assert_ne!(DIMENSION, 0);
        let mut m = Matrix::<T, DIMENSION, DIMENSION>::new();
        for row in 0..DIMENSION {
            m.set(row, row, T::one());
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Add<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds two matrices together
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Add<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds a scalar to a matrix
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> AddAssign<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a matrix to a matrix and stores the result in the first matrix
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> AddAssign<T>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a scalar to a matrix and stores the result in the first matrix
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Sub<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method subtracts two matrices
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> SubAssign<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method subtracts two matrices
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method multiplies a matrix with a scalar
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
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> MulAssign<T>
    for Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method multiplies a matrix with a scalar
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
    Mul<Matrix<T, ROWS, OTHERS_COLS>> for Matrix<T, OWN_COLS, ROWS>
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
// Implements indexing for the matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for Matrix<T, ROWS, COLS>
{
    type Output = T;
    /// Allows for array like access to the matrix
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
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        self.get_mut(index.0, index.1)
    }
}

// ================================================================
// Tests
// ================================================================
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
}

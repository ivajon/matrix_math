pub mod helpers;
pub use helpers::*;

use crate::traits::{CompliantNumerical, MatrixInterface};
use core::{ops::*, usize};

#[derive(Debug, Clone, PartialEq)]
/// # Matrix
/// This is a generic matrix struct, it is used to represent a matrix of any numeric type,
/// it is generic in the type of the matrix, and it is generic in the number of rows and columns
/// of the matrix.
pub struct Matrix<T: CompliantNumerical, const ROWS: usize, const COLS: usize> {
    elements: [[T; COLS]; ROWS],
}

/// Simpler alias for a 2x2 matrix
pub type Matrix2x2<T> = Matrix<T, 2, 2>;
/// Simpler alias for a 3x3 matrix
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
/// Simpler alias for a 4x4 matrix
pub type Matrix4x4<T> = Matrix<T, 4, 4>;

// Special case
impl<T: CompliantNumerical, const COLS: usize> From<[T; COLS]> for Matrix<T, 1, COLS>
where
    Self: Clone,
{
    fn from(value: [T; COLS]) -> Self {
        [value].into()
    }
}

impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]>
    for Matrix<T, ROWS, COLS>
where
    Self: Clone,
{
    fn from(value: [[T; COLS]; ROWS]) -> Self {
        Self::new_from_data(value)
    }
}

impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> From<Matrix<T, ROWS, COLS>>
    for [[T; COLS]; ROWS]
{
    fn from(val: Matrix<T, ROWS, COLS>) -> Self {
        val.elements
    }
}

impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> MatrixInterface<T, ROWS, COLS>
    for Matrix<T, ROWS, COLS>
where
    Self: Clone,
{
    /// Creates a new matrix of the specified size
    fn new() -> Matrix<T, ROWS, COLS> {
        let elements: [[T; COLS]; ROWS] =
            array_init::array_init(|_| array_init::array_init(|_| T::default()));
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
    fn set_elements(&mut self, data: [[T; COLS]; ROWS]) {
        self.elements = data;
    }
    // ================================================================
    // Convience functions
    // ================================================================
    fn transpose(&mut self) -> Matrix<T, COLS, ROWS> {
        let mut m: Matrix<_, COLS, ROWS> = Matrix::<T, COLS, ROWS>::new();
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

// impl<T: CompliantNumerical, const N: usize> Matrix<T, N, N> {
//     pub fn determinant(&self) -> T {
//         todo!()
//     }
// }

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

impl<
        T: CompliantNumerical + Add<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > Add<Matrix<TOther, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    fn add(mut self, other: Matrix<TOther, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        for (this, other) in self.elements.iter_mut().zip(other.elements.iter()) {
            for (this, other) in this.iter_mut().zip(other.iter()) {
                *this = this.clone() + other.clone()
            }
        }
        self
    }
}
// Implementing the add method for the generic matrix struct and a scalar
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Add<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn add(mut self, other: T) -> Matrix<T, ROWS, COLS> {
        for this in self.elements.iter_mut() {
            for this in this.iter_mut() {
                *this += other.clone()
            }
        }
        self
    }
}

// Allows for add assign to be used on a matrix
impl<
        T: CompliantNumerical + Add<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > AddAssign<Matrix<TOther, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    fn add_assign(&mut self, other: Matrix<TOther, ROWS, COLS>) {
        for (this, other) in self.elements.iter_mut().zip(other.elements.iter()) {
            for (this, other) in this.iter_mut().zip(other.iter()) {
                *this = this.clone() + other.clone()
            }
        }
    }
}

impl<
        T: CompliantNumerical + AddAssign<TOther>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > AddAssign<TOther> for Matrix<T, ROWS, COLS>
{
    fn add_assign(&mut self, other: TOther) {
        for row in self.elements.iter_mut() {
            for el in row.iter_mut() {
                *el += other.clone()
            }
        }
    }
}

impl<
        T: CompliantNumerical + Sub<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > Sub<Matrix<TOther, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, COLS>;
    fn sub(mut self, other: Matrix<TOther, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        for (this, other) in self.elements.iter_mut().zip(other.elements.iter()) {
            for (this, other) in this.iter_mut().zip(other.iter()) {
                // This is a bit ugly but it might be needed
                *this = this.clone() - other.clone();
            }
        }
        self
    }
}

impl<
        T: CompliantNumerical + SubAssign<TOther>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const COLS: usize,
    > SubAssign<Matrix<TOther, ROWS, COLS>> for Matrix<T, ROWS, COLS>
{
    fn sub_assign(&mut self, other: Matrix<TOther, ROWS, COLS>) {
        let other = other.elements;
        for (this, other) in self.elements.iter_mut().zip(other.iter()) {
            for (this, other) in this.iter_mut().zip(other.iter()) {
                *this -= other.clone();
            }
        }
    }
}

impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn mul(mut self, other: T) -> Matrix<T, ROWS, COLS> {
        for row in self.elements.iter_mut() {
            for el in row.iter_mut() {
                // We cannot get rid of this clone without unsafe behaviour
                *el *= other.clone();
            }
        }
        self
    }
}
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> MulAssign<T>
    for Matrix<T, ROWS, COLS>
{
    fn mul_assign(&mut self, other: T) {
        for row in self.elements.iter_mut() {
            for el in row.iter_mut() {
                // We cannot get rid of this clone without unsafe behaviour
                *el *= other.clone();
            }
        }
    }
}

impl<
        T: CompliantNumerical + Mul<TOther, Output = T>,
        TOther: CompliantNumerical,
        const ROWS: usize,
        const OWN_COLS: usize,
        const OTHERS_COLS: usize,
    > Mul<Matrix<TOther, ROWS, OTHERS_COLS>> for Matrix<T, OWN_COLS, ROWS>
{
    type Output = Matrix<T, OWN_COLS, OTHERS_COLS>;
    #[inline(never)]
    fn mul(self, other: Matrix<TOther, ROWS, OTHERS_COLS>) -> Matrix<T, OWN_COLS, OTHERS_COLS> {
        // This is horendous
        let mut result = Matrix::new();
        for row in 0..OWN_COLS {
            for col in 0..OTHERS_COLS {
                let mut sum = T::default();
                for i in 0..ROWS {
                    sum += self.get(row, i).clone() * other.get(i, col).clone();
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
        self.get(index.0, index.1)
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

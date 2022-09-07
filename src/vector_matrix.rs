//! Defines a matrix
use core::{
    marker::PhantomData,
    ops::{self, Index, IndexMut},
};

#[allow(unused_macros)]
macro_rules! CONVERT_TO_SIZE {
    ($numel:expr) => {
        ($numel / 4 + ((($numel % 4) != 0) as u32)) as usize
    };
}
use crate::{
    traits::{CompliantNumerical, VectorTrait},
    x86_optimization::*,
    MatrixInterface,
};
pub struct X86Matrix<T: CompliantNumerical, const ROWS: usize, const COLS: usize> {
    _t: PhantomData<T>, // T is used in elements
    elements: [X86Vector<COLS, { CONVERT_TO_SIZE!(COLS) }>; ROWS], // Holds elements in
}

/// Simpeler alias for a 2x2 matrix
pub type X86Matrix2x2<T> = X86Matrix<T, 2, 2>;
/// Simpeler alias for a 3x3 matrix
pub type X86Matrix3x3<T> = X86Matrix<T, 3, 3>;
/// Simpeler alias for a 4x4 matrix
pub type X86Matrix4x4<T> = X86Matrix<T, 4, 4>;

#[allow(dead_code)]
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> MatrixInterface<T, ROWS, COLS>
    for X86Matrix<T, ROWS, COLS>
{
    /// Creates a new matrix of the specified size
    fn new() -> X86Matrix<T, ROWS, COLS> {
        let elements = [[T::default(); COLS]; ROWS];
        X86Matrix {
            _t: PhantomData,
            elements,
        }
    }
    /// Instantiantes a new matrix with the given elements
    fn new_from_data(elements: [[T; COLS]; ROWS]) -> X86Matrix<T, ROWS, COLS> {
        X86Matrix {
            _t: PhantomData,
            elements,
        }
    }
    // ================================================================
    // Getters
    // ================================================================
    /// Get the value of the element at the given row and column
    fn get(&self, row: usize, col: usize) -> &T {
        &self.elements[row].get(col)
    }
    /// Get the mutable value of the element at the given row and column
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m = X86Matrix::<f32, 2, 2>::new();
    /// let mut val = m.get_mut(0, 0);
    /// *val = 1.0;
    /// ```
    /// # Mathematical equivalent
    /// ```latex
    /// A_{ij} = 1
    /// ```
    #[inline(always)]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.elements[row].get_mut(col)
    }
    // ================================================================
    // Setters
    // ================================================================
    /// Sets the value of a given element in the matrix
    fn set(&mut self, row: usize, col: usize, value: T) {
        self.elements[row].set(col, value);
    }

    /// Sets the entire elements array
    /// This is useful for when you want to do something with the entire matrix
    /// without having to use the get and set functions
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m = X86Matrix::<f32, 2, 2>::new();
    /// m.set_elements([[1.0, 2.0], [3.0, 4.0]]);
    /// ```

    fn set_elements(&mut self, elements: [[T; COLS]; ROWS]) {
        for (index, row) in self.elements.iter_mut().enumerate() {
            *row = X86Vector::new_from_data(elements[index]);
        }
    }
    // ================================================================
    // Convience functions
    // ================================================================

    /// Passes every element of the matrix to a function defined as
    /// ```rust
    /// fn f(x: f64) -> f64 {
    ///     x.exp()
    /// }
    /// ```

    fn map<F>(&self, f: F) -> X86Matrix<T, ROWS, COLS>
    where
        F: Fn(T) -> T,
    {
        let mut m = Self::new();
        for (index, vec) in self.elements.iter_mut().enumerate() {
            m.elements[index] = vec.map(f);
        }
        m
    }

    fn identity() -> Self {
        todo!()
    }

    type TransposeOutput = Self;

    fn transpose(&mut self) -> Self::TransposeOutput {
        todo!()
    }

    fn zeros() -> Self {
        Self::new()
    }

    fn rows(&self) -> usize {
        ROWS
    }

    fn cols(&self) -> usize {
        COLS
    }

    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn eye() -> Self {
        Self::identity()
    }
}
// ================================================================
// X86Matrix Operations implementations
// ================================================================
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> X86Matrix<T, ROWS, COLS> {
    /// Returns the identity matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let m = X86Matrix::<f64, 2, 2>::eye();
    /// assert_eq!(*m.get_elements(), [[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    pub fn eye() -> X86Matrix<T, ROWS, COLS> {
        let mut m = X86Matrix::<T, ROWS, COLS>::new();
        for row in 0..ROWS {
            m.set(row, row, T::one());
        }
        m
    }
    /// Just an alias for the eye function
    pub fn identity() -> X86Matrix<T, ROWS, COLS> {
        X86Matrix::eye()
    }
    /// Returns a matrix with the same data but each vector represents a column and not a row
    fn rotate_vectors<ROWVEC>(&mut self) -> X86Matrix<T, COLS, ROWS>
    where
        ROWVEC: VectorTrait<T, ROWS>,
    {
        let mut m = X86Matrix::<T, COLS, ROWS>::new();
        for col in 0..COLS {
            let mut v = VEC::new();
            for row in 0..ROWS {
                v.set(row, self.elements[row][col].clone());
            }
            m.set_vector(col, v);
        }
        m
    }
}
#[allow(dead_code)]
/// Defines a add method for the generic matrix struct
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Add<X86Matrix<T, ROWS, COLS>>
    for X86Matrix<T, ROWS, COLS>
{
    type Output = X86Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds two matrices together
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let m1 = X86Matrix::<f32, 2, 2>::new();
    /// let m2 = X86Matrix::<f32, 2, 2>::new();
    /// let m3 = m1 + m2;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
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
    fn add(self, other: X86Matrix<T, ROWS, COLS>) -> X86Matrix<T, ROWS, COLS> {
        let mut result: VectorTrait<T, COLS> = X86Matrix::new();
        for (index, (own_vec, other_vec)) in
            self.elements.iter().zip(other.elements.iter()).enumerate()
        {
            result.elements[index] = *own_vec + *other_vec;
        }
        result
    }
}
// Implementing the add method for the generic matrix struct and a scalar
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Add<T>
    for X86Matrix<T, ROWS, COLS>
{
    type Output = X86Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method adds a scalar to a matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let m1 = X86Matrix::<f32, 2, 2>::new();
    /// let m2 = m1 + 1.0;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[1.0, 1.0],
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
    fn add(self, other: T) -> X86Matrix<T, ROWS, COLS> {
        let mut result = X86Matrix::new();
        let scalar_vec = vec![other; ROWS];
        for (index, vec) in self.elements.iter().enumerate() {
            result.elements[index] = *vec + scalar_vec;
        }
        result
    }
}

// Allows for add assign to be used on a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::AddAssign<X86Matrix<T, ROWS, COLS>> for X86Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a matrix to a matrix and stores the result in the first matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// let m2 = X86Matrix::<f32, 2, 2>::new();
    /// m1 += m2;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
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
    fn add_assign(&mut self, other: X86Matrix<T, ROWS, COLS>) {
        for (own_vec, other_vec) in self.elements.iter_mut().zip(other.elements.iter()) {
            *own_vec += *other_vec;
        }
    }
}

// Allows for add assign to be used on a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::AddAssign<T>
    for X86Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method adds a scalar to a matrix and stores the result in the first matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// m1 += 1.0;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[1.0, 1.0],
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
        // Get a constant vector of the same size as the matrix columns
        let scalarvec = vec![other; COLS];
        // Add this vector to the matrix
        for vec in self.elements.iter_mut() {
            *vec += scalarvec;
        }
    }
}

#[allow(dead_code)]
/// Defines a sub method for the generic matrix struct
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Sub<X86Matrix<T, ROWS, COLS>>
    for X86Matrix<T, ROWS, COLS>
{
    type Output = X86Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method subtracts two matrices
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let m1 = X86Matrix::<f32, 2, 2>::new();
    /// let m2 = X86Matrix::<f32, 2, 2>::new();
    /// let m3 = m1 - m2;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
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

    fn sub(self, other: X86Matrix<T, ROWS, COLS>) -> X86Matrix<T, ROWS, COLS> {
        let mut result = X86Matrix::new();
        for (index, (vec, othervec)) in self.elements.iter().zip(other.elements.iter()).enumerate()
        {
            result.elements[index] = *vec - *othervec;
        }
        result
    }
}
// Implements sub assign for a matrix and a matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize>
    ops::SubAssign<X86Matrix<T, ROWS, COLS>> for X86Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method subtracts two matrices
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// let m2 = X86Matrix::<f32, 2, 2>::new();
    /// m1 -= m2;
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
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
    fn sub_assign(&mut self, other: X86Matrix<T, ROWS, COLS>) {
        for (vec, othervec) in self.elements.iter_mut().zip(other.elements.iter()) {
            *vec -= *othervec;
        }
    }
}
// Implements multiplication for a matrix and a scalar
#[allow(dead_code)]
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> ops::Mul<T>
    for X86Matrix<T, ROWS, COLS>
{
    type Output = X86Matrix<T, ROWS, COLS>;
    /// # What is this?
    /// This method multiplies a matrix with a scalar
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// m1[(1,1)] = 2.0;
    /// let m2 = m1 * 2.0;
    /// assert_eq!(m2[(1,1)], 4.0);
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
    ///     [0.0, 2.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    fn mul(self, other: T) -> X86Matrix<T, ROWS, COLS> {
        let mut result = X86Matrix::new();
        let scalarvec = vec![other; COLS];
        for (index, vec) in self.elements.iter().enumerate() {
            result.elements[index] = *vec * scalarvec;
        }
        result
    }
}
// Implements mul assign
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> core::ops::MulAssign<T>
    for X86Matrix<T, ROWS, COLS>
{
    /// # What is this?
    /// This method multiplies a matrix with a scalar
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// m1[(1,1)] = 2.0;
    /// m1 *= 2.0;
    /// assert_eq!(m1[(1,1)], 4.0);
    /// ```
    /// # Output
    /// ```text
    /// X86Matrix([[0.0, 0.0],
    ///     [0.0, 2.0]])
    /// ```
    /// # Panics
    /// This method never panics
    /// # Safety
    /// This method is safe
    fn mul_assign(&mut self, other: T) {
        let scalarvec = vec![other; COLS];
        for vec in &mut self.elements {
            *vec *= scalarvec;
        }
    }
}

// Multiplying an integer matrix with a float matrix is nonsensical, so it is not implemented
// If you want to do that first convert the integer matrix to a float matrix or vice versa
impl<T: CompliantNumerical, const ROWS: usize, const OWN_COLS: usize, const OTHERS_COLS: usize>
    core::ops::Mul<X86Matrix<T, ROWS, OTHERS_COLS>> for X86Matrix<T, OWN_COLS, ROWS>
{
    type Output = X86Matrix<T, OWN_COLS, OTHERS_COLS>;
    #[inline(never)]
    /// # What is this?
    /// This is the implementation for the multiplication of a matrix with another matrix
    /// # Why should I use this?
    /// This implementation gaurantees that the dimensions of the matrices are correct, so that the
    /// multiplication can be done without any errors. This falls directly from the fact
    /// that the matricies are statically sized, so the compiler can check the dimensions
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m1 = X86Matrix::<f32, 2, 2>::new();
    /// let mut m2 = X86Matrix::<f32, 2, 2>::new();
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
    fn mul(self, other: X86Matrix<T, ROWS, OTHERS_COLS>) -> X86Matrix<T, OWN_COLS, OTHERS_COLS> {
        let mut result = X86Matrix::new();

        let mut temp = other.rotate_vectors();
        for (row, own_vec) in self.elements.iter().enumerate() {
            for (col, other_vec) in temp.elements.iter_mut().enumerate() {
                temp[(row, col)] = (*other_vec).dot(own_vec);
            }
        }
        result
    }
}
// Implements indexing for the matrix
impl<T: CompliantNumerical, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for X86Matrix<T, ROWS, COLS>
{
    type Output = T;
    /// Allows for array like access to the matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let m = X86Matrix::<f32, 2, 2>::new();
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
    for X86Matrix<T, ROWS, COLS>
{
    /// Allows for array like assignment to the matrix
    /// # Example
    /// ```rust
    /// use matrs::matrix::X86Matrix;
    /// let mut m = X86Matrix::<f32, 2, 2>::new();
    /// m[(0, 1)] = 1.0;
    /// ```
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        self.get_mut(index.0, index.1)
    }
}

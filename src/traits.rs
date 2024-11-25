use core::marker::PhantomData;
use core::ops::*;
use num::traits::{Num, NumAssign, NumAssignOps, NumOps, One, Zero};

use crate::Vector;
/// Defines a compliant numerical trait
/// It is used as a generic to only allow numbers that can be casted to/ from floats
/// and integers
/// This trait is used to ensure that the matrix struct can only be used with numbers
pub trait CompliantNumerical:
    One + Zero + Num + NumOps + NumAssign + NumAssignOps + Sized + Clone + Default
{
    fn sqrt(num: Self) -> Self;
}
impl CompliantNumerical for f32 {
    // FROM https://github.com/emkw/rust-fast_inv_sqrt
    fn sqrt(num: Self) -> Self {
        let i = num.to_bits();
        let i = 0x5f3759df - (i >> 1);
        let y = f32::from_bits(i);

        1f32 / (y * (1.5 - 0.5 * num * y * y))
    }
}
impl CompliantNumerical for f64 {
    // FROM https://github.com/emkw/rust-fast_inv_sqrt
    fn sqrt(num: Self) -> Self {
        // Magic number based on Chris Lomont work:
        const MAGIC_U64: u64 = 0x5fe6ec85e7de30da;
        const THREEHALFS: f64 = 1.5;
        let x2 = num * 0.5;
        let i = MAGIC_U64 - (num.to_bits() >> 1);
        let y: f64 = f64::from_bits(i);

        1f64 / (y * (THREEHALFS - (x2 * y * y)))
    }
}
impl CompliantNumerical for i8 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for i16 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for i32 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for i64 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for u8 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for u16 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for u32 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for u64 {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}
impl CompliantNumerical for usize {
    fn sqrt(num: Self) -> Self {
        num::integer::sqrt(num)
    }
}

pub trait MatrixInterface<T: CompliantNumerical, const ROWS: usize, const COLS: usize>:
    Add<T>
    + AddAssign<T>
    + Mul<T>
    + MulAssign<T>
    + Index<(usize, usize)>
    + IndexMut<(usize, usize)>
    + Sized
    + From<[[T; COLS]; ROWS]>
    + Into<[[T; COLS]; ROWS]>
{
    /// Creates a new matrix of the specified size
    fn new() -> Self;

    /// Creates a new matrix of zeros of the specified size
    fn zeros() -> Self {
        Self::new()
    }

    /// Instantiantes a new matrix with the given elements
    fn new_from_data(data: [[T; COLS]; ROWS]) -> Self;

    /// Get the value of the element at the given row and column
    fn get(&self, row: usize, col: usize) -> &T;

    /// Get the mutable value of the element at the given row and column
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T;

    /// Returns the number of rows in the matrix
    #[inline(always)]
    fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns in the matrix
    #[inline(always)]
    fn cols(&self) -> usize {
        COLS
    }

    /// Returns the size of the matrix
    #[inline(always)]
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    // ================================================================
    // Setters
    // ================================================================
    /// Sets the value of a given element in the matrix
    fn set(&mut self, row: usize, col: usize, value: T);

    /// Sets the entire elements array
    /// This is useful for when you want to do something with the entire matrix
    /// without having to use the get and set functions
    fn set_elements(&mut self, data: [[T; COLS]; ROWS]);

    // ================================================================
    // Convience functions
    // ================================================================

    type TransposeOutput;

    /// Transposes a given matrix, since we can't assume the matrix to be square, this function takes O(n) memory
    /// and takes O(n^2) time to transpose a matrix, if the matrix is square this could be done in O(1) memory and theta(n^2 / 2) time, which is similar but better
    fn transpose(&mut self) -> Self::TransposeOutput;

    /// Passes every element of the matrix to a function defined as
    fn map<F: Fn(T) -> T>(&self, f: F) -> Self;

    /// Returns the identity matrix
    fn eye() -> Self {
        Self::identity()
    }

    /// Just an alias for the eye function
    fn identity() -> Self;

    /// Gauss eliminates against the rhs vector.
    /// This is a bit inefficient.
    fn gauss(&self, rhs: &Vector<T, ROWS>) -> Vector<T, COLS>
    where
        Self: Clone,
        T: core::fmt::Debug,
    {
        let mut intermediate = self.clone();
        let mut res = rhs.clone();
        let mut values: Vector<T, COLS> = Vector::new();
        for col in 0..COLS {
            let upper_row = self.get_row(col);
            for row in (col + 1)..ROWS {
                if *self.get(row, col) == T::default() {
                    continue;
                }
                let upper_factor = self.get(row, col).clone();
                let lower_factor = self.get(col, col).clone();
                let upper_row = upper_row.clone() * upper_factor.clone();
                let lower_row = self.get_row(row) * lower_factor.clone();

                intermediate.set_row(row, lower_row - upper_row);

                res[row] = res[row].clone() * lower_factor - res[col].clone() * upper_factor;
            }
        }
        values[ROWS - 1] = res[0].clone() / intermediate.get(ROWS - 1, COLS - 1).clone();
        // Assume that it has been reduced in to a stair case shape. This should be the case.
        for row in (0..(ROWS - 1)).rev() {
            let mut sub = T::default();
            // Compute the constants to be subtracted.
            // axn = k - bxn-1 - cxn-2...
            for col in row..ROWS {
                sub += intermediate.get(row, col).clone() * values[col].clone();
            }
            let factor = intermediate.get(row, row).clone();
            values[row] = (res[row].clone() - sub) / factor;
        }

        values
    }

    /// Returns the specified row.
    ///
    /// ## Panics.
    ///
    /// This function panics if the row does not exist.
    fn get_row(&self, row: usize) -> Vector<T, COLS>;
    /// Returns the specified col.
    ///
    /// ## Panics
    ///
    /// This function panics if the column does not exist.
    fn get_col(&self, col: usize) -> Vector<T, ROWS>;

    /// Sets the Specified row.
    ///
    /// ## Panics.
    ///
    /// This function panics if the row does not exist.
    fn set_row(&mut self, row: usize, vals: Vector<T, COLS>);

    /// Sets the specified col.
    ///
    /// ## Panics
    ///
    /// This function panics if the column does not exist.
    fn set_col(&mut self, col: usize, vals: Vector<T, ROWS>);
}

/// Creating an api for vectors, this api includes the general instantiation and
/// some operations, it's easy to see that cross products are not implemented, that's
/// the case because it does not really make sense for vectors outside R3
pub trait VectorTrait<T: CompliantNumerical, const SIZE: usize> {
    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    fn new() -> Self;

    /// Creates a new Vector with all elements set to the given value
    fn new_from_value(val: T) -> Self;

    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    fn new_from_data(data: [T; SIZE]) -> Self;

    fn set(&mut self, index: usize, value: T);

    /// Gets the element at the specified index
    /// # Panics
    /// On index out of bounds
    fn get(&self, index: usize) -> T;

    /// Constant access to the vector
    fn get_const<const INDEX: usize>(&self) -> T;

    /// Returns a mutable refference to the given index
    fn get_mut(&mut self, index: usize) -> &mut T;

    /// Returns a mutable refference to the given index
    /// using a constant index
    fn get_mut_const<const INDEX: usize>(&mut self) -> &mut T;

    /// Element wise multiplication, and summarization of the result
    fn dot(&self, other: &Self) -> T;

    /// Returns the magnitude of the vector in the
    /// same dataformat the the vector is represented in
    fn magnitude(&self) -> T;

    /// Converts the vector to a length one vector with the same direction
    fn normalize(&mut self);

    /// Adds two vectors with one another
    fn add(&self, other: &Self) -> Self;

    /// Subtracts to vectors from one another
    fn sub(&self, other: &Self) -> Self;

    /// Element wise multiplication
    fn elementwise_mul(&self, other: &Self) -> Self;

    /// Element wise division of two vectors
    fn div(&self, other: &Self) -> Self;

    /// Adds two vectors and stores the result in self
    fn add_assign(&mut self, other: &Self);

    /// Subtracts two vectors and stores the result in self
    fn sub_assign(&mut self, other: &Self);

    /// Multiplies two vectors and stores the result in self
    fn mul_assign(&mut self, other: &Self);

    /// Divides two vectors and stores the result in self
    fn div_assign(&mut self, other: &Self);

    /// Computs the sum of a vector
    fn sum(&self) -> T;

    fn iter(&mut self) -> VectorItterator<T, Self, SIZE>
    where
        Self: Sized;
}

pub struct VectorItterator<
    'a,
    T: CompliantNumerical,
    VectorType: VectorTrait<T, COUNT>,
    const COUNT: usize,
> {
    pub(crate) vec: &'a mut VectorType,
    pub(crate) t: PhantomData<T>,
    pub(crate) count: usize,
}

impl<'a, T: CompliantNumerical, VectorType: VectorTrait<T, COUNT>, const COUNT: usize> Iterator
    for VectorItterator<'a, T, VectorType, COUNT>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < COUNT {
            self.count += 1;
            return Some(self.vec.get(self.count - 1));
        }
        None
    }
}

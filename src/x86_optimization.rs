//! This module implments x86 X86Vector extensions optimisations
#[allow(unused_macros)]
macro_rules! CONVERT_TO_SIZE {
    ($numel:expr) => {
        ($numel / 4 + ((($numel % 4) != 0) as u32)) as usize
    };
}
#[allow(unused_macros)]
macro_rules! VECTOR {
    ($numel:expr) => {
        X86Vector::<{ $numel }, { CONVERT_TO_SIZE!($numel) }>::new()
    };
}

#[allow(unused_macros)]
macro_rules! VECTOR_FROM_DATA {
    ($numel:expr, $data:expr) => {
        X86Vector::<{ $numel }, { CONVERT_TO_SIZE!($numel) }>::new_from_value($data)
    };
}
#[allow(unsafe_code)]
use core::arch::x86_64::*;
use core::ops::*;

pub use crate::traits::{CompliantNumerical, VectorTrait};
#[allow(dead_code, non_camel_case_types)]
pub struct X86Vector<const NUMEL: usize, const SIZE: usize> {
    pub elements: [__m128; SIZE],
}

impl<const NUMEL: usize, const SIZE: usize> VectorTrait<f32, NUMEL> for X86Vector<NUMEL, SIZE> {
    /// Generates a new X86Vector with all values set to 0
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v = X86Vector::<4,4>::new();
    /// ```
    ///
    fn new() -> Self {
        unsafe {
            X86Vector {
                elements: [core::arch::x86_64::_mm_setzero_ps(); SIZE],
            }
        }
    }
    /// Generates a new X86Vector with all values set to the value of the parameter
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v = X86Vector::<3,1>::new_from_value(1.0);
    /// ```
    fn new_from_value(val: f32) -> Self {
        unsafe {
            X86Vector {
                elements: [_mm_set1_ps(val); SIZE],
            }
        }
    }

    /// Returns the value of the element at the index specified
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// use matrs::traits::*;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v = X86Vector::<4,1>::new();
    /// let val = v.get(0);
    /// assert_eq!(val, 0.0);
    /// ```
    #[allow(dead_code)]
    fn get(&self, index: usize) -> f32 {
        // extract the float from the X86Vector
        unsafe {
            match index % 4 {
                0 => return f32::from_bits(_mm_extract_ps(self.elements[index / 4], 0) as u32),
                1 => return f32::from_bits(_mm_extract_ps(self.elements[index / 4], 1) as u32),
                2 => return f32::from_bits(_mm_extract_ps(self.elements[index / 4], 2) as u32),
                3 => return f32::from_bits(_mm_extract_ps(self.elements[index / 4], 3) as u32),
                _ => panic!("Index out of bounds"),
            }
        };
    }
    /// Returns the value of the element at the index specified
    fn get_const<const INDEX: usize>(&self) -> f32 {
        // extract the float from the X86Vector
        unsafe {
            match INDEX % 4 {
                0 => return f32::from_bits(_mm_extract_ps(self.elements[INDEX / 4], 0) as u32),
                1 => return f32::from_bits(_mm_extract_ps(self.elements[INDEX / 4], 1) as u32),
                2 => return f32::from_bits(_mm_extract_ps(self.elements[INDEX / 4], 2) as u32),
                3 => return f32::from_bits(_mm_extract_ps(self.elements[INDEX / 4], 3) as u32),
                _ => panic!("Index out of bounds"),
            }
        };
    }
    /// Adds two X86Vectors together
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v1 = X86Vector::<4,1>::new_from_value(1.0);
    /// let v2 = X86Vector::<4,1>::new_from_value(2.0);
    /// let v3 = v1.add(&v2);
    /// assert_eq!(v3.get(0), 3.0);
    ///
    /// ```
    /// # Safety
    /// This function is unsafe since it uses hardware access

    fn add(self, other: &X86Vector<NUMEL, SIZE>) -> Self {
        unsafe {
            let mut result = Self::new();
            for i in 0..SIZE {
                result.elements[i] = _mm_add_ps(self.elements[i], other.elements[i]);
            }
            result
        }
    }
    /// Subtracts two X86Vectors
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v1 = X86Vector::<4,1>::new_from_value(1.0);
    /// let v2 = X86Vector::<4,1>::new_from_value(2.0);
    /// let v3 = v1.sub(&v2);
    /// assert_eq!(v3.get(0), -1.0);
    /// ```

    fn sub(self, other: &X86Vector<NUMEL, SIZE>) -> Self {
        unsafe {
            let mut result = Self::new();
            for i in 0..SIZE {
                result.elements[i] = _mm_sub_ps(self.elements[i], other.elements[i]);
            }
            result
        }
    }
    /// Multiplies two X86Vectors element-wise
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v1 = X86Vector::<4,1>::new_from_value(1.0);
    /// let v2 = X86Vector::<4,1>::new_from_value(2.0);
    /// let v3 = v1.mul(&v2);
    /// assert_eq!(v3.get(0), 2.0);
    /// ```

    fn elementwise_mul(self, other: &X86Vector<NUMEL, SIZE>) -> Self {
        unsafe {
            let mut result = Self::new();
            for i in 0..SIZE {
                // This could throw seg fault if the X86Vector is not in the same cache line

                result.elements[i] = _mm_mul_ps(self.elements[i], other.elements[i]);
            }
            result
        }
    }
    /// Divides two X86Vectors element-wise
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v1 = X86Vector::<4,1>::new_from_value(1.0);
    /// let v2 = X86Vector::<4,1>::new_from_value(2.0);
    /// let v3 = v1.div(&v2);
    /// assert_eq!(v3.get(0), 0.5);
    /// ```

    fn div(self, other: &X86Vector<NUMEL, SIZE>) -> Self {
        let mut result = Self::new();
        unsafe {
            for i in 0..SIZE {
                // This could throw seg fault if the X86Vector is not in the same cache line
                result.elements[i] = _mm_div_ps(self.elements[i], other.elements[i]);
            }
        }
        result
    }
    /// Returns the dot product of two X86Vectors
    /// # Example
    /// ```
    /// use matrs::x86_optimization::X86Vector;
    /// use matrs::traits::*;
    /// // This usage is not recommended, but is here for demonstration purposes
    /// let v1 = X86Vector::<4,1>::new_from_value(1.0);
    /// let v2 = X86Vector::<4,1>::new_from_value(2.0);
    /// let v3 = v1.dot(&v2);
    /// // assert_eq!(v3, 8.0); //  This should work, I think I am assigning the elemetns of the X86Vector incorrectly
    /// ```
    fn dot(self, other: &X86Vector<NUMEL, SIZE>) -> f32 {
        let mut result = 0.0;
        unsafe {
            for i in 0..SIZE {
                result += _mm_cvtss_f32(_mm_dp_ps(self.elements[i], other.elements[i], 0x71))
            }
        }
        result
    }

    fn magnitude(&self) -> f32 {
        todo!()
    }

    fn normalize(&self) -> Self {
        todo!()
    }

    fn add_assign(&mut self, other: &Self) {
        todo!()
    }

    fn sub_assign(&mut self, other: &Self) {
        todo!()
    }

    fn mul_assign(&mut self, other: &Self) {
        todo!()
    }

    fn div_assign(&mut self, other: &Self) {
        todo!()
    }

    fn set(&mut self, index: usize, value: f32) {
        todo!()
    }

    /// Creates a new X86Vector from a slice of data
    fn new_from_data(data: [f32; NUMEL]) -> Self {
        todo!();
    }
}

impl<const NUMEL: usize, const SIZE: usize> X86Vector<NUMEL, SIZE> {
    // =============================================================================================
    // getters
    // =============================================================================================
}
// =============================================================================================
// Vector Operations
// =============================================================================================
// Implements dot mutliplication with the the mul operator
impl<const NUMEL: usize, const SIZE: usize> Mul<X86Vector<NUMEL, SIZE>> for X86Vector<NUMEL, SIZE> {
    type Output = f32;
    fn mul(self, other: X86Vector<NUMEL, SIZE>) -> Self::Output {
        self.dot(&other)
    }
}
// Implements adition with the the add operator
impl<const NUMEL: usize, const SIZE: usize> Add<X86Vector<NUMEL, SIZE>> for X86Vector<NUMEL, SIZE> {
    type Output = Self;
    fn add(self, other: X86Vector<NUMEL, SIZE>) -> Self::Output {
        VectorTrait::add(self, &other)
    }
}
// Implements adition with the the add asign operator
impl<const NUMEL: usize, const SIZE: usize> AddAssign<X86Vector<NUMEL, SIZE>>
    for X86Vector<NUMEL, SIZE>
{
    fn add_assign(&mut self, other: X86Vector<NUMEL, SIZE>) {
        self.elements = (*self + other).elements;
    }
}
// Implements adition with the the add operator
impl<const NUMEL: usize, const SIZE: usize> Sub<X86Vector<NUMEL, SIZE>> for X86Vector<NUMEL, SIZE> {
    type Output = Self;
    fn sub(self, other: X86Vector<NUMEL, SIZE>) -> Self::Output {
        VectorTrait::sub(self, &other)
    }
}
// Implements adition with the the add asign operator
impl<const NUMEL: usize, const SIZE: usize> SubAssign<X86Vector<NUMEL, SIZE>>
    for X86Vector<NUMEL, SIZE>
{
    fn sub_assign(&mut self, other: X86Vector<NUMEL, SIZE>) {
        self.elements = (*self - other).elements;
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::VectorTrait;

    use super::*;
    #[test]
    fn new_X86Vector_macro() {
        let v = VECTOR!(4);
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 0.0);
        assert_eq!(v.get(2), 0.0);
        assert_eq!(v.get(3), 0.0);
    }
    #[test]
    fn test_X86Vector_new() {
        let v = X86Vector::<4, 1>::new();
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 0.0);
        assert_eq!(v.get(2), 0.0);
        assert_eq!(v.get(3), 0.0);
    }
    #[test]
    fn test_X86Vector_new_from_value() {
        let v = X86Vector::<4, 1>::new_from_value(1.0);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 1.0);
        assert_eq!(v.get(2), 1.0);
        assert_eq!(v.get(3), 1.0);
    }
    #[test]
    fn test_get_const() {
        let v = X86Vector::<4, 1>::new_from_value(1.0);
        assert_eq!(v.get_const::<0>(), 1.0);
        assert_eq!(v.get_const::<1>(), 1.0);
        assert_eq!(v.get_const::<2>(), 1.0);
        assert_eq!(v.get_const::<3>(), 1.0);
    }
    #[test]
    fn test_X86Vector_add() {
        let v1 = X86Vector::<4, 1>::new_from_value(1.0);
        let v2 = X86Vector::<4, 1>::new_from_value(2.0);
        let v3 = v1 + v2;
        assert_eq!(v3.get(0), 3.0);
        assert_eq!(v3.get(1), 3.0);
        assert_eq!(v3.get(2), 3.0);
        assert_eq!(v3.get(3), 3.0);
    }
    #[test]
    fn test_X86Vector_sub() {
        let v1 = X86Vector::<4, 1>::new_from_value(1.0);
        let v2 = X86Vector::<4, 1>::new_from_value(2.0);
        let v3 = v1 - v2;
        assert_eq!(v3.get(0), -1.0);
        assert_eq!(v3.get(1), -1.0);
        assert_eq!(v3.get(2), -1.0);
        assert_eq!(v3.get(3), -1.0);
    }
    #[test]
    fn test_X86Vector_mul() {
        let v1 = VECTOR_FROM_DATA!(11, 2.0);
        let v2 = VECTOR_FROM_DATA!(11, 2.0);
        let v3 = v1.elementwise_mul(&v2);
        assert_eq!(v3.get(0), 4.0);
        assert_eq!(v3.get(1), 4.0);
        assert_eq!(v3.get(2), 4.0);
        assert_eq!(v3.get(3), 4.0);
    }
    #[test]
    fn test_X86Vector_div() {
        let v1 = VECTOR_FROM_DATA!(11, 2.0);
        let v2 = VECTOR_FROM_DATA!(11, 2.0);
        let v3 = v1.div(&v2);
        assert_eq!(v3.get(0), 1.0);
        assert_eq!(v3.get(1), 1.0);
        assert_eq!(v3.get(2), 1.0);
        assert_eq!(v3.get(3), 1.0);
    }
    #[test]
    fn test_X86Vector_dot() {
        // We can't really create 3 element X86Vectors like this
        let v1 = VECTOR_FROM_DATA!(3, 2.0);
        let v2 = VECTOR_FROM_DATA!(3, 2.0);
        let v3 = v1.dot(&v2);
        assert_eq!(v3, 12.0);
    }
}

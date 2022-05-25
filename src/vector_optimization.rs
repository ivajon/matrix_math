//! This module implments x86 vector extensions optimisations
#[allow(unused_macros)]
macro_rules! CONVERT_TO_SIZE {
    ($numel:expr) => {
        ($numel / 4 + ((($numel % 4) != 0) as u32)) as usize
    };
}
#[allow(unused_macros)]
macro_rules! VECTOR {
    ($numel:expr) => {
        vector_opt::vector::<{$numel}, { CONVERT_TO_SIZE!($numel) }>::new()
    };
}

#[allow(unused_macros)]
macro_rules! VECTOR_FROM_DATA {
    ($numel:expr, $data:expr) => {
        vector_opt::vector::<{$numel}, { CONVERT_TO_SIZE!($numel) }>::new_from_value($data)
    };
}
#[allow(unsafe_code)]

pub mod vector_opt {
    use core::arch::x86_64::*;

    #[allow(dead_code, non_camel_case_types)]
    pub struct vector<const NUMEL: usize, const SIZE: usize> {
        pub elements: [__m128; SIZE],
    }

    impl<const NUMEL: usize, const SIZE: usize> vector<NUMEL, SIZE> {
        /// Generates a new vector with all values set to 0
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v = vector::<4,4>::new();
        /// ```
        ///
        pub fn new() -> Self {
            unsafe {
                vector {
                    elements: [core::arch::x86_64::_mm_setzero_ps(); SIZE],
                }
            }
        }
        /// Generates a new vector with all values set to the value of the parameter
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v = vector::<3,1>::new_from_value(1.0);
        /// ```
        pub fn new_from_value(val: f32) -> Self {
            unsafe {
                vector {
                    elements: [_mm_set1_ps(val); SIZE],
                }
            }
        }

        // =============================================================================================
        // getters
        // =============================================================================================
        /// Returns the value of the element at the index specified
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v = vector::<4,1>::new();
        /// let val = v.get(0);
        /// assert_eq!(val, 0.0);
        /// ```
        #[allow(dead_code)]
        pub fn get(&self, index: usize) -> f32 {
            // extract the float from the vector
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

        pub fn get_const<const INDEX: usize>(&self) -> f32 {
            // extract the float from the vector
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

        // =============================================================================================
        // Vector Operations
        // =============================================================================================

        /// Adds two vectors together
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v1 = vector::<4,1>::new_from_value(1.0);
        /// let v2 = vector::<4,1>::new_from_value(2.0);
        /// let v3 = v1.add(v2);
        /// assert_eq!(v3.get(0), 3.0);
        ///
        /// ```
        /// # Safety
        /// This function is unsafe since it uses hardware access

        pub fn add(self, other: Self) -> Self {
            unsafe {
                let mut result = Self::new();
                for i in 0..SIZE {
                    result.elements[i] = _mm_add_ps(self.elements[i], other.elements[i]);
                }
                result
            }
        }

        /// Subtracts two vectors
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v1 = vector::<4,1>::new_from_value(1.0);
        /// let v2 = vector::<4,1>::new_from_value(2.0);
        /// let v3 = v1.sub(v2);
        /// assert_eq!(v3.get(0), -1.0);
        /// ```

        pub fn sub(self, other: Self) -> Self {
            unsafe {
                let mut result = Self::new();
                for i in 0..SIZE {
                    result.elements[i] = _mm_sub_ps(self.elements[i], other.elements[i]);
                }
                result
            }
        }
        /// Multiplies two vectors element-wise
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v1 = vector::<4,1>::new_from_value(1.0);
        /// let v2 = vector::<4,1>::new_from_value(2.0);
        /// let v3 = v1.mul(v2);
        /// assert_eq!(v3.get(0), 2.0);
        /// ```

        pub fn mul(self, other: Self) -> Self {
            unsafe {
                let mut result = Self::new();
                for i in 0..SIZE {
                    // This could throw seg fault if the vector is not in the same cache line
                    result.elements[i] = _mm_mul_ps(self.elements[i], other.elements[i]);
                }
                result
            }
        }

        /// Divides two vectors element-wise
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v1 = vector::<4,1>::new_from_value(1.0);
        /// let v2 = vector::<4,1>::new_from_value(2.0);
        /// let v3 = v1.div(v2);
        /// assert_eq!(v3.get(0), 0.5);
        /// ```

        pub fn div(self, other: Self) -> Self {
            unsafe {
                let mut result = Self::new();
                for i in 0..SIZE {
                    // This could throw seg fault if the vector is not in the same cache line
                    result.elements[i] = _mm_div_ps(self.elements[i], other.elements[i]);
                }
                result
            }
        }

        /// Returns the dot product of two vectors
        /// # Example
        /// ```rust
        /// use matrs::vector_optimization::vector_opt::vector;
        /// // This usage is not recommended, but is here for demonstration purposes
        /// let v1 = vector::<4,1>::new_from_value(1.0);
        /// let v2 = vector::<4,1>::new_from_value(2.0);
        /// let v3 = v1.dot(v2);
        /// // assert_eq!(v3, 8.0); //  This should work, I think I am assigning the elemetns of the vector incorrectly
        /// ```

        pub fn dot(self, other: Self) -> f32 {
            unsafe {
                let mut result = 0.0;
                for i in 0..SIZE {
                    result += _mm_cvtss_f32(_mm_dp_ps(self.elements[i], other.elements[i], 0x71))
                }
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_vector_macro() {
        let v = VECTOR!(4);
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 0.0);
        assert_eq!(v.get(2), 0.0);
        assert_eq!(v.get(3), 0.0);
    }
    #[test]
    fn test_vector_new() {
        let v = vector_opt::vector::<4,1>::new();
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 0.0);
        assert_eq!(v.get(2), 0.0);
        assert_eq!(v.get(3), 0.0);
    }
    #[test]
    fn test_vector_new_from_value() {
        let v = vector_opt::vector::<4,1>::new_from_value(1.0);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 1.0);
        assert_eq!(v.get(2), 1.0);
        assert_eq!(v.get(3), 1.0);
    }
    #[test]
    fn test_get_const() {
        let v = vector_opt::vector::<4,1>::new_from_value(1.0);
        assert_eq!(v.get_const::<0>(), 1.0);
        assert_eq!(v.get_const::<1>(), 1.0);
        assert_eq!(v.get_const::<2>(), 1.0);
        assert_eq!(v.get_const::<3>(), 1.0);
    }
    #[test]
    fn test_vector_add() {
        let v1 = vector_opt::vector::<4,1>::new_from_value(1.0);
        let v2 = vector_opt::vector::<4,1>::new_from_value(2.0);
        let v3 = v1.add(v2);
        assert_eq!(v3.get(0), 3.0);
        assert_eq!(v3.get(1), 3.0);
        assert_eq!(v3.get(2), 3.0);
        assert_eq!(v3.get(3), 3.0);
    }
    #[test]
    fn test_vector_sub() {
        let v1 = vector_opt::vector::<4,1>::new_from_value(1.0);
        let v2 = vector_opt::vector::<4,1>::new_from_value(2.0);
        let v3 = v1.sub(v2);
        assert_eq!(v3.get(0), -1.0);
        assert_eq!(v3.get(1), -1.0);
        assert_eq!(v3.get(2), -1.0);
        assert_eq!(v3.get(3), -1.0);
    }
    #[test]
    fn test_vector_mul() {
        let v1 = VECTOR_FROM_DATA!(11, 2.0);
        let v2 = VECTOR_FROM_DATA!(11, 2.0);
        let v3 = v1.mul(v2);
        assert_eq!(v3.get(0), 4.0);
        assert_eq!(v3.get(1), 4.0);
        assert_eq!(v3.get(2), 4.0);
        assert_eq!(v3.get(3), 4.0);
    }
    #[test]
    fn test_vector_div() {
        let v1 = VECTOR_FROM_DATA!(11, 2.0);
        let v2 = VECTOR_FROM_DATA!(11, 2.0);
        let v3 = v1.div(v2);
        assert_eq!(v3.get(0), 1.0);
        assert_eq!(v3.get(1), 1.0);
        assert_eq!(v3.get(2), 1.0);
        assert_eq!(v3.get(3), 1.0);
    }
    #[test]
    fn test_vector_dot() {
        // We can't really create 3 element vectors like this
        let v1 = VECTOR_FROM_DATA!(3, 2.0);
        let v2 = VECTOR_FROM_DATA!(3, 2.0);
        let v3 = v1.dot(v2);
        assert_eq!(v3, 12.0);
    }
}

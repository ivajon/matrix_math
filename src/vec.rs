//! A Vector is of n elements, and is stored in a contiguous block of memory.
//!
//! # Supported operations:
//!
//! * Indexing
//! * Slicing
//! * Iteration
//! * Addition
//! * Subtraction
//! * Multiplication
//! * Division by scalar
//! * Dot product
//! * Cross product
//! # Example
//! ```rust
//! use matrs::vec::Vec;
//! let mut v = Vec::<f32, 3>::new();
//! v.set(0, 1.0);
//! v.set(1, 2.0);
//! v.set(2, 3.0);
//! assert_eq!(*v.get(0), 1.0);
//! assert_eq!(*v.get(1), 2.0);
//! assert_eq!(*v.get(2), 3.0);
//! ```
use crate::traits;
use std::{
    ops::{Div, Mul},
    usize,
};

/// Type definition for a Vector
/// It is a fixed size Vector
/// It is used to implement linear algebra for neural nets and similar
/// # Benefits
/// * It is a fixed size Vector
/// * It gaurantees that the Vector to Vector operations are of correct size
///
/// # Example
/// ```rust
/// use matrs::vec::Vec;
/// let v1 = Vec::<f32, 3>::new();
/// let v2 = Vec::<f32, 3>::new();
/// let v3 = v1 + v2;
/// ```
/// # Future work
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec<T: traits::CompliantNumerical, const COUNT: usize> {
    elements: [T; COUNT],
}
pub type Vec2<T> = Vec<T, 2>;
pub type Vec3<T> = Vec<T, 3>;
pub type Vec4<T> = Vec<T, 4>;
#[allow(dead_code)]
// Implements a new method for the generic Vector struct
impl<T: traits::CompliantNumerical, const COUNT: usize> Vec<T, COUNT> {
    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let Vec = Vec::< f64, 3 >::new();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to create a new Vector
    pub fn new() -> Vec<T, COUNT> {
        let elements = [T::default(); COUNT];
        Vec { elements }
    }

    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let data:[f32;3] = [1.0, 2.0, 3.0];
    /// let Vec = Vec::new_from_data(data);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to create a new Vector
    pub fn new_from_data(data: [T; COUNT]) -> Vec<T, COUNT> {
        Vec { elements: data }
    }
    /// Gets the element at the specified index
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let Vec = Vec::< f64, 3 >::new();
    /// let element = Vec.get(1);
    /// ```
    /// # Panics
    /// On index out of bounds
    /// # Safety
    /// It is safe if the index is within the bounds of the Vector
    pub fn get(&self, index: usize) -> &T {
        &self.elements[index]
    }
    /// Sets the element at the specified index
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// Vec.set(1, 1.0);
    /// let mut element = Vec.get(1);
    /// ```
    /// # Panics
    /// On index out of bounds
    /// # Safety
    /// It is safe if the index is within the bounds of the Vector
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }
    /// Sets the element at the specified index
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// Vec.set(1, 1.0);
    /// ```
    /// # Panics
    /// On index out of bounds
    /// # Safety
    /// It is safe if the index is within the bounds of the Vector
    pub fn set(&mut self, index: usize, value: T) {
        self.elements[index] = value;
    }
    /// Gets the length of the Vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let Vec = Vec::< f64, 3 >::new();
    /// let length = Vec.len();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to get the length of the Vector
    /// # Note
    /// The length of the Vector is always the same as the amount of elements
    /// in the Vector
    pub fn len(&self) -> usize {
        COUNT
    }
    /// Convert the Vector to a slice
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// let slice = Vec.iter();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert the Vector to a slice
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.elements.iter()
    }
    /// Convert the Vector to a slice
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// let slice = Vec.iter_mut();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert the Vector to a slice
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.elements.iter_mut()
    }
    /// Passes each element of the Vector to the function
    pub fn for_each(&self, f: T)
    where
        T: Fn(T) -> T,
    {
        for element in self.elements.iter() {
            f(*element);
        }
    }
    pub fn cross_mult(&self, other: Vec<T, COUNT>, ret: &mut Vec<T, COUNT>) {
        if COUNT == 3 {
            ret.set(
                0,
                self.get(1).clone() * other.get(2).clone()
                    - self.get(2).clone() * other.get(1).clone(),
            );
            ret.set(
                1,
                self.get(2).clone() * other.get(0).clone()
                    - self.get(0).clone() * other.get(2).clone(),
            );
            ret.set(
                2,
                self.get(0).clone() * other.get(1).clone()
                    - self.get(1).clone() * other.get(0).clone(),
            );
        }
    }
    /// Mutliplies two vectors element by element, and returns the result in the vector ret
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// let mut other = Vec::< f64, 3 >::new();
    /// let mut ret = Vec::< f64, 3 >::new();
    /// Vec.set(0, 1.0);
    /// Vec.set(1, 2.0);
    /// Vec.set(2, 3.0);
    /// other.set(0, 4.0);
    /// other.set(1, 5.0);
    /// other.set(2, 6.0);
    /// Vec.element_wise_mult(other, &mut ret);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to multiply two vectors
    /// # Note
    /// The length of the Vector is always the same as the amount of elements
    pub fn element_wise_mult(&self, other: Vec<T, COUNT>, ret: &mut Vec<T, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self.get(i).clone() * other.get(i).clone());
        }
    }
    /// Divides two vectors element by element, and returns the result in the vector ret
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut Vec = Vec::< f64, 3 >::new();
    /// let mut other = Vec::< f64, 3 >::new();
    /// let mut ret = Vec::< f64, 3 >::new();
    /// Vec.set(0, 1.0);
    /// Vec.set(1, 2.0);
    /// Vec.set(2, 3.0);
    /// other.set(0, 4.0);
    /// other.set(1, 5.0);
    /// other.set(2, 6.0);
    /// Vec.element_wise_div(other, &mut ret);
    /// ```
    /// # Panics
    /// Panics if element of other is 0
    /// # Safety
    /// It is not safe to divide two vectors since it's possible to divide by 0
    /// # Note
    /// The length of the Vector is always the same as the amount of elements
    pub fn element_wise_div(&self, other: Vec<T, COUNT>, ret: &mut Vec<T, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self.get(i).clone() / other.get(i).clone());
        }
    }
    /// Converts an integer vector to a f32 vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut a = Vec::< i32, 3 >::new();
    /// let mut b = Vec::< f32, 3 >::new();
    /// a.to_f32(&mut b);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert an integer vector to a f32 vector

    pub fn to_f32(&self, ret: &mut Vec<f32, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self.get(i).clone().into_f32());
        }
    }
    /// Converts an integer vector to a f64 vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let mut a = Vec::< i32, 3 >::new();
    /// let mut b = Vec::< f64, 3 >::new();
    /// a.to_f64(&mut b);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert an integer vector to a f64 vector
    /// # Note
    /// The length of the Vector is always the same as the amount of elements

    pub fn to_f64(&self, ret: &mut Vec<f64, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self.get(i).clone().into_f64());
        }
    }
}

// Multiplies a Vector with a scalar
impl<T: traits::CompliantNumerical, const COUNT: usize> Mul<T> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    fn mul(self, other: T) -> Vec<T, COUNT> {
        let mut ret = Vec::new();
        for i in 0..COUNT {
            ret.set(i, *self.get(i) * other);
        }
        ret
    }
}
// Divides a Vector with a scalar
impl<T: traits::CompliantNumerical, const COUNT: usize> Div<T> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    /// Divides a Vector with a scalar
    /// # Panics
    /// Panics if the divisor is 0
    /// # Safety
    /// This function is unsafe because it can cause undefined behavior, but so is all division
    fn div(self, other: T) -> Vec<T, COUNT> {
        if other == T::default() {
            panic!("Division by 0");
        }
        let mut ret = Vec::new();
        for i in 0..COUNT {
            ret.set(i, *self.get(i) / other);
        }
        ret
    }
}

// Implements Vector addition
impl<T: traits::CompliantNumerical, const COUNT: usize> std::ops::Add<Vec<T, COUNT>>
    for Vec<T, COUNT>
{
    type Output = Vec<T, COUNT>;
    /// This function adds two Vectors
    /// Since Vectors are allocated at compile time this implmenetation gauraantees that the
    /// resulting Vector has the same size as the first Vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vec;
    /// let a = Vec::<f32,3>::new_from_data([1.0, 2.0, 3.0]);
    /// let b = Vec::<f32,3>::new_from_data([4.0, 5.0, 6.0]);
    /// let c = a + b;
    /// assert_eq!(c.len(), 3);
    /// assert_eq!(*c.get(0), 5.0);
    /// assert_eq!(*c.get(1), 7.0);
    /// assert_eq!(*c.get(2), 9.0);
    /// ```
    /// # Panics
    /// This function never panics
    /// # Safety
    /// This function is safe
    fn add(self, other: Vec<T, COUNT>) -> Vec<T, COUNT> {
        let mut result = Vec::new();
        for i in 0..COUNT {
            result.set(i, self.get(i).clone() + other.get(i).clone());
        }
        result
    }
}

// Implements Vector subtraction
impl<T: traits::CompliantNumerical, const COUNT: usize> std::ops::Sub<Vec<T, COUNT>>
    for Vec<T, COUNT>
{
    type Output = Vec<T, COUNT>;
    fn sub(self, other: Vec<T, COUNT>) -> Vec<T, COUNT> {
        let mut result = Vec::new();
        for i in 0..COUNT {
            result.set(i, self.get(i).clone() - other.get(i).clone());
        }
        result
    }
}

// Implements Vector dotproduct
impl<T: traits::CompliantNumerical, const COUNT: usize> std::ops::Mul<Vec<T, COUNT>>
    for Vec<T, COUNT>
{
    type Output = Vec<T, COUNT>;
    fn mul(self, other: Vec<T, COUNT>) -> Vec<T, COUNT> {
        let mut result = Vec::new();
        for i in 0..COUNT {
            result.set(i, self.get(i).clone() * other.get(i).clone());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_Vec_add() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(*c.get(0), 5.0, "Add did not work. Wrong value at index 0 ");
        assert_eq!(*c.get(1), 7.0, "Add did not work. Wrong value at index 1");
        assert_eq!(*c.get(2), 9.0, "Add did not work. Wrong value at index 2");
    }
    #[test]
    fn test_Vec_sub() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a - b;
        assert_eq!(
            *c.get(0),
            -3.0,
            "Sub did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(1),
            -3.0,
            "Sub did not work. Wrong value at index 1, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(2),
            -3.0,
            "Sub did not work. Wrong value at index 2, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
    #[test]
    fn test_Vec_mul() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(
            *c.get(0),
            4.0,
            "Mul did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(1),
            10.0,
            "Mul did not work. Wrong value at index 1, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(2),
            18.0,
            "Mul did not work. Wrong value at index 2, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
    #[test]
    fn test_Vec_div() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = 2.0;
        let c = a / b;
        assert_eq!(
            *c.get(0),
            1.0 / 2.0,
            "Div did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(1),
            2.0 / 2.0,
            "Div did not work. Wrong value at index 1, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(2),
            3.0 / 2.0,
            "Div did not work. Wrong value at index 2, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
    #[test]
    fn test_Vec_cross_mult() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let mut c = Vec::new_from_data([0.0, 0.0, 0.0]);
        a.cross_mult(b, &mut c);
        assert_eq!(
            *c.get(0),
            -3.0,
            "Cross mult did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(1),
            6.0,
            "Cross mult did not work. Wrong value at index 1, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(2),
            -3.0,
            "Cross mult did not work. Wrong value at index 2, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
    #[test]
    fn test_Vec_dot_mult() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(
            *c.get(0),
            4.0,
            "Dot mult did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(1),
            10.0,
            "Dot mult did not work. Wrong value at index 1, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
        assert_eq!(
            *c.get(2),
            18.0,
            "Dot mult did not work. Wrong value at index 2, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
}

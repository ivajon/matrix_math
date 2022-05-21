use crate::traits::CompliantNumerical;
///! Defines a static memory vector struct
///! It has a user defined amount of elements
///! It is used to implement linear algebra for neural nets and similar
use std::{
    ops::{Div, Mul},
    usize,
};

/// Type definition for a vector
/// It is a fixed size vector
/// It is used to implement linear algebra for neural nets and similar
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec<T: CompliantNumerical, const COUNT: usize> {
    elements: [T; COUNT],
}
#[allow(dead_code)]
// Implements a new method for the generic vector struct
impl<T: CompliantNumerical, const COUNT: usize> Vec<T, COUNT> {
    /// Creates a new vector
    pub fn new() -> Vec<T, COUNT> {
        let elements = [T::default(); COUNT];
        Vec { elements }
    }

    /// Creates a new vector with the given elements
    pub fn new_from_data(data: [T; COUNT]) -> Vec<T, COUNT> {
        Vec {
            elements: data, //[T::default();COUNT],
        }
    }
    pub fn get(&self, index: usize) -> &T {
        &self.elements[index]
    }
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }
    pub fn set(&mut self, index: usize, value: T) {
        self.elements[index] = value;
    }
    pub fn len(&self) -> usize {
        COUNT
    }
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.elements.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.elements.iter_mut()
    }

    // Passes every element of the vector to the given function
    pub fn for_each(&self, f: T)
    where
        T: Fn(T),
    {
        for element in self.elements.iter() {
            f(*element);
        }
    }
    fn cross_mult(self, other: Vec<T, COUNT>, ret: &mut Vec<T, COUNT>) {
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
}

// Multiplies a vector with a scalar
impl<T: CompliantNumerical, const COUNT: usize> Mul<T> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    fn mul(self, other: T) -> Vec<T, COUNT> {
        let mut ret = Vec::new();
        for i in 0..COUNT {
            ret.set(i, *self.get(i) * other);
        }
        ret
    }
}
// Divides a vector with a scalar
impl<T: CompliantNumerical, const COUNT: usize> Div<T> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    fn div(self, other: T) -> Vec<T, COUNT> {
        let mut ret = Vec::new();
        for i in 0..COUNT {
            ret.set(i, *self.get(i) / other);
        }
        ret
    }
}

// Implements vector addition
impl<T: CompliantNumerical, const COUNT: usize> std::ops::Add<Vec<T, COUNT>> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    fn add(self, other: Vec<T, COUNT>) -> Vec<T, COUNT> {
        let mut result = Vec::new();
        for i in 0..COUNT {
            result.set(i, self.get(i).clone() + other.get(i).clone());
        }
        result
    }
}

// Implements vector subtraction
impl<T: CompliantNumerical, const COUNT: usize> std::ops::Sub<Vec<T, COUNT>> for Vec<T, COUNT> {
    type Output = Vec<T, COUNT>;
    fn sub(self, other: Vec<T, COUNT>) -> Vec<T, COUNT> {
        let mut result = Vec::new();
        for i in 0..COUNT {
            result.set(i, self.get(i).clone() - other.get(i).clone());
        }
        result
    }
}

// Implements vector dotproduct
impl<T: CompliantNumerical, const COUNT: usize> std::ops::Mul<Vec<T, COUNT>> for Vec<T, COUNT> {
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
    fn test_vec_add() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(*c.get(0), 5.0);
        assert_eq!(*c.get(1), 7.0);
        assert_eq!(*c.get(2), 9.0);
    }
    #[test]
    fn test_vec_sub() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a - b;
        assert_eq!(*c.get(0), -3.0);
        assert_eq!(*c.get(1), -3.0);
        assert_eq!(*c.get(2), -3.0);
    }
    #[test]
    fn test_vec_mul() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(*c.get(0), 4.0);
        assert_eq!(*c.get(1), 10.0);
        assert_eq!(*c.get(2), 18.0);
    }
    #[test]
    fn test_vec_div() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = 2.0;
        let c = a / b;
        assert_eq!(*c.get(0), 1.0 / 2.0);
        assert_eq!(*c.get(1), 2.0 / 2.0);
        assert_eq!(*c.get(2), 3.0 / 2.0);
    }
    #[test]
    fn test_vec_cross_mult() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let mut c = Vec::new_from_data([0.0, 0.0, 0.0]);
        a.cross_mult(b, &mut c);
        assert_eq!(*c.get(0), -3.0);
        assert_eq!(*c.get(1), 6.0);
        assert_eq!(*c.get(2), -3.0);
    }
    #[test]
    fn test_vec_dot_mult() {
        let a = Vec::new_from_data([1.0, 2.0, 3.0]);
        let b = Vec::new_from_data([4.0, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(*c.get(0), 4.0);
        assert_eq!(*c.get(1), 10.0);
        assert_eq!(*c.get(2), 18.0);
    }
}

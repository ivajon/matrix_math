use num_traits::NumCast;

use crate::traits::VectorItterator;
use crate::CompliantNumerical;
use crate::VectorTrait;
use core::{ops::*, usize};

/// Type definition for a Vector
///
/// # Example
/// ```rust
/// use matrs::vec::Vector;
/// let v1 = Vector::<f32, 3>::new();
/// let v2 = Vector::<f32, 3>::new();
/// let v3 = v1 + v2;
/// ```
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<T: CompliantNumerical, const COUNT: usize> {
    elements: [T; COUNT],
}
pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;

impl<T: CompliantNumerical, const COUNT: usize> VectorTrait<T, COUNT> for Vector<T, COUNT> {
    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let Vec = Vector::< f64, 3 >::new();
    /// ```
    fn new() -> Vector<T, COUNT> {
        let elements = array_init::array_init(|_| T::default());
        Vector { elements }
    }

    fn new_from_value(val: T) -> Vector<T, COUNT> {
        Vector {
            elements: array_init::array_init(|_| val.clone()),
        }
    }

    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let data:[f32;3] = [1.0, 2.0, 3.0];
    /// let Vec = Vector::new_from_data(data);
    /// ```
    fn new_from_data(data: [T; COUNT]) -> Vector<T, COUNT> {
        Vector { elements: data }
    }
    /// Sets the element at the specified index
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut Vec = Vector::< f64, 3 >::new();
    /// Vec.set(1, 1.0);
    /// ```
    fn set(&mut self, index: usize, value: T) {
        self.elements[index] = value;
    }

    /// Gets a copy of the element at the specified index
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let Vec = Vector::< f64, 3 >::new();
    /// let element = Vec.get(1);
    /// ```
    fn get(&self, index: usize) -> T {
        self.elements[index].clone()
    }

    /// Gets a copy of the element at the specified index
    fn get_const<const INDEX: usize>(&self) -> T {
        self.elements[INDEX].clone()
    }
    /// Gets the scalar product of 2 vectors
    fn dot(&self, other: &Self) -> T {
        self.clone() * other.clone()
    }

    fn magnitude(&self) -> T {
        let mut sum = T::default();
        for el in self.iter() {
            sum += el.clone() * el.clone();
        }
        T::sqrt(sum)
    }

    fn normalize(&mut self) {
        let magnitude = self.magnitude();
        for el in self.elements.iter_mut() {
            *el = el.clone() / magnitude.clone();
        }
    }

    fn add(&self, other: &Self) -> Self {
        let mut arr = array_init::array_init(|_| T::zero());
        for index in 0..COUNT {
            arr[index] = self[index].clone() + other[index].clone();
        }
        Vector::new_from_data(arr)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut arr = array_init::array_init(|_| T::zero());
        for index in 0..COUNT {
            arr[index] = self[index].clone() - other[index].clone();
        }
        Vector::new_from_data(arr)
    }

    fn elementwise_mul(&self, other: &Self) -> Self {
        let mut arr = array_init::array_init(|_| T::zero());
        for index in 0..COUNT {
            arr[index] = self[index].clone() * other[index].clone();
        }
        Vector::new_from_data(arr)
    }

    fn div(&self, other: &Self) -> Self {
        let mut arr = array_init::array_init(|_| T::zero());
        for index in 0..COUNT {
            arr[index] = self[index].clone() / other[index].clone();
        }
        Vector::new_from_data(arr)
    }

    fn add_assign(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(this, other)| *this += other.clone());
    }

    fn sub_assign(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(this, other)| *this -= other.clone());
    }

    fn mul_assign(&mut self, other: &Self) {
        self.element_wise_mult(other.clone());
    }

    fn div_assign(&mut self, other: &Self) {
        self.element_wise_div(other.clone());
    }

    fn iter(&mut self) -> VectorItterator<T, Self, COUNT> {
        VectorItterator {
            vec: self,
            t: core::marker::PhantomData,
            count: 0,
        }
    }

    fn sum(&self) -> T {
        let mut sum = T::zero();
        for el in self.iter() {
            sum += el.clone();
        }
        sum
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }

    fn get_mut_const<const INDEX: usize>(&mut self) -> &mut T {
        &mut self.elements[INDEX]
    }
}

#[allow(dead_code)]
// Implements a new method for the generic Vector struct
impl<T: CompliantNumerical, const COUNT: usize> Default for Vector<T, COUNT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CompliantNumerical, const COUNT: usize> Vector<T, COUNT> {
    /// Creates a new Vector
    /// It is used to create a new Vector with a user defined amount of elements
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let Vec = Vector::< f64, 3 >::new();
    /// ```
    pub fn new() -> Vector<T, COUNT> {
        let elements = array_init::array_init(|_| T::default());
        Vector { elements }
    }

    /// Creates a new Vector
    pub fn new_from_data(data: [T; COUNT]) -> Vector<T, COUNT> {
        Vector { elements: data }
    }
    // ================================================================
    // Getters
    // ================================================================

    /// Gets the element at the specified index
    pub fn get(&self, index: usize) -> &T {
        &self.elements[index]
    }
    /// Sets the element at the specified index
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }

    /// Gets the entire Vector elements as arr
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let Vec = Vector::< f64, 3 >::new();
    /// let arr = Vec.get_elements();
    /// ```
    pub fn get_elements(&self) -> &[T; COUNT] {
        &self.elements
    }
    /// Gets the dimension of the Vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let Vec = Vector::< f64, 3 >::new();
    /// let length = Vec.size();
    /// ```
    pub const fn size(&self) -> usize {
        COUNT
    }
    // ================================================================
    // Setters
    // ================================================================
    /// Sets the element at the specified index
    pub fn set(&mut self, index: usize, value: T) {
        self.elements[index] = value;
    }

    /// Passes each element of the Vector to the function
    pub fn for_each(&self, f: T)
    where
        T: Fn(&T),
    {
        self.elements.iter().for_each(f);
    }

    /// Multiplies two vectors element by element
    pub fn element_wise_mult(&mut self, other: Vector<T, COUNT>) {
        self.iter_mut()
            .zip(other.elements)
            .for_each(|(this, other)| *this *= other);
    }

    /// Divides two vectors element by element
    pub fn element_wise_div(&mut self, other: Vector<T, COUNT>) {
        self.iter_mut()
            .zip(other.elements)
            .for_each(|(this, other)| *this /= other);
    }

    // ================================================================
    // Convenience operators
    // ================================================================
    /// Calculates the length of a vector in a n dimensional space
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::< f64, 3 >::new();
    /// a.set(0, 1.0);
    /// a.set(1, 2.0);
    /// a.set(2, 3.0);
    /// let length = a.length();
    /// ```
    pub fn length(&self) -> T {
        T::sqrt(self.clone() * self.clone())
    }

    /// Passes every element of the Vec to a function defined as
    /// ```rust
    /// fn f(x: &mut f64) {
    ///
    ///     *x = x.exp();
    /// }
    /// ```
    pub fn map(&mut self, f: T)
    where
        T: Fn(&mut T),
    {
        self.elements.iter_mut().for_each(f);
    }
    /// Normalizes a vector in a n dimensional space
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::< f64, 3 >::new();
    /// a.set(0, 1.0);
    /// a.set(1, 2.0);
    /// a.set(2, 3.0);
    /// let b = a.normalize();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to normalize a vector
    pub fn normalize(self) -> Vector<T, COUNT> {
        let len = self.length();
        self / len
    }
    // ================================================================
    // Converters
    // ================================================================
    /// Convert the Vector to a slice
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut Vec = Vector::< f64, 3 >::new();
    /// let slice = Vec.iter();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert the Vector to a slice
    pub fn iter(&self) -> core::slice::Iter<T> {
        self.elements.iter()
    }
    /// Convert the Vector to a slice
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut Vec = Vector::< f64, 3 >::new();
    /// let slice = Vec.iter_mut();
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert the Vector to a slice
    pub fn iter_mut(&mut self) -> core::slice::IterMut<T> {
        self.elements.iter_mut()
    }
}

impl<T: CompliantNumerical + NumCast, const COUNT: usize> Vector<T, COUNT> {
    /// Converts an integer vector to a f32 vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::< i32, 3 >::new();
    /// let mut b = Vector::< f32, 3 >::new();
    /// a.to_f32(&mut b);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert an integer vector to a f32 vector

    pub fn to_f32(&self, ret: &mut Vector<f32, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self[i].clone().to_f32().unwrap());
        }
    }

    /// Converts an integer vector to a f64 vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::< i32, 3 >::new();
    /// let mut b = Vector::< f64, 3 >::new();
    /// a.to_f64(&mut b);
    /// ```
    /// # Panics
    /// Never panics
    /// # Safety
    /// It is safe to convert an integer vector to a f64 vector
    /// # Note
    /// The length of the Vector is always the same as the amount of elements
    pub fn to_f64(&self, ret: &mut Vector<f64, COUNT>) {
        for i in 0..COUNT {
            ret.set(i, self[i].clone().to_f64().unwrap());
        }
    }
}

// ================================================================
// Implementations
// ================================================================
// Multiplies a Vector with a scalar
impl<T: CompliantNumerical, const COUNT: usize> Mul<T> for Vector<T, COUNT> {
    type Output = Vector<T, COUNT>;
    fn mul(mut self, other: T) -> Vector<T, COUNT> {
        self *= other;
        self
    }
}
// Mul assign a Vector with a scalar
impl<T: CompliantNumerical, const COUNT: usize> MulAssign<T> for Vector<T, COUNT> {
    /// Implements the *= operator
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::< f64, 3 >::new();
    /// a.set(0, 1.0);
    /// a.set(1, 2.0);
    /// a.set(2, 3.0);
    /// a *= 2.0;
    /// assert_eq!(a[0], 2.0);
    /// ```
    fn mul_assign(&mut self, other: T) {
        self.elements
            .iter_mut()
            .for_each(|this| *this *= other.clone());
    }
}
// Divides a Vector with a scalar
impl<T: CompliantNumerical + DivAssign<TOther>, TOther: CompliantNumerical, const COUNT: usize>
    Div<TOther> for Vector<T, COUNT>
{
    type Output = Vector<T, COUNT>;
    fn div(mut self, other: TOther) -> Vector<T, COUNT> {
        self.elements
            .iter_mut()
            .for_each(|this| *this /= other.clone());
        self
    }
}

// Implements Vector addition
impl<T: CompliantNumerical + Add<T, Output = T>, const COUNT: usize> Add<Vector<T, COUNT>>
    for Vector<T, COUNT>
{
    type Output = Vector<T, COUNT>;
    fn add(mut self, other: Vector<T, COUNT>) -> Vector<T, COUNT> {
        self.elements
            .iter_mut()
            .zip(other.elements)
            .for_each(|(this, other)| *this += other);
        self
    }
}

// Implements Vector subtraction
impl<T: CompliantNumerical, const COUNT: usize> Sub<Vector<T, COUNT>> for Vector<T, COUNT> {
    type Output = Vector<T, COUNT>;
    fn sub(mut self, other: Vector<T, COUNT>) -> Vector<T, COUNT> {
        self.elements
            .iter_mut()
            .zip(other.elements)
            .for_each(|(this, other)| *this -= other);
        self
    }
}

impl<T: CompliantNumerical, const COUNT: usize> Mul<Vector<T, COUNT>> for Vector<T, COUNT> {
    type Output = T;
    fn mul(self, other: Vector<T, COUNT>) -> T {
        let mut collector: T = T::zero();
        for el in self
            .elements
            .into_iter()
            .zip(other.elements)
            .map(|(this, other)| this * other)
        {
            collector += el;
        }
        collector
    }
}

impl<T: CompliantNumerical, const COUNT: usize> Index<usize> for Vector<T, COUNT> {
    type Output = T;
    /// Allows for array like access to a vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let a = Vector::<f32,3>::new_from_data([1.0, 2.0, 3.0]);
    /// assert_eq!(a[0], 1.0);
    /// assert_eq!(a[1], 2.0);
    /// assert_eq!(a[2], 3.0);
    /// ```
    fn index(&self, index: usize) -> &T {
        self.get(index)
    }
}

// Implements mutable indexing
impl<T: CompliantNumerical, const COUNT: usize> IndexMut<usize> for Vector<T, COUNT> {
    /// Allows for array like access to a vector
    /// # Example
    /// ```rust
    /// use matrs::vec::Vector;
    /// let mut a = Vector::<f32,3>::new_from_data([1.0, 2.0, 3.0]);
    /// a[0] = 2.0;
    /// assert_eq!(a[0], 2.0);
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index)
    }
}
impl<T: CompliantNumerical> Vector3<T> {
    /// Defines the cross product of 2 vectors
    /// # Example
    /// ```rust
    /// use matrs::vec::{Vector,Vector3};
    /// let a : Vector3<f64> = Vector::new_from_data([1.0, 2.0, 3.0]);
    /// let b : Vector3<f64> = Vector::new_from_data([4.0, 5.0, 6.0]);
    /// let c = a.cross(b);
    /// assert_eq!(c[0], -3.0);
    /// assert_eq!(c[1], 6.0);
    /// assert_eq!(c[2], -3.0);
    /// ```

    pub fn cross(self, other: Vector3<T>) -> Vector3<T> {
        let data = [
            self[1].clone() * other[2].clone() - self[2].clone() * other[1].clone(),
            self[2].clone() * other[0].clone() - self[0].clone() * other[2].clone(),
            self[0].clone() * other[1].clone() - self[1].clone() * other[0].clone(),
        ];
        Vector3 { elements: data }
    }
    pub fn dot(self, other: Vector3<T>) -> T {
        let mut collector = T::zero();
        for el in self
            .elements
            .iter()
            .zip(other.elements)
            .map(|(this, other)| this.clone() * other)
        {
            collector += el;
        }
        collector
    }
}

// ================================================================
// Tests
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vec_add() {
        let a = Vector::new_from_data([1.0, 2.0, 3.0]);
        let b = Vector::new_from_data([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c[0], 5.0, "Add did not work. Wrong value at index 0 ");
        assert_eq!(c[1], 7.0, "Add did not work. Wrong value at index 1");
        assert_eq!(c[2], 9.0, "Add did not work. Wrong value at index 2");
    }
    #[test]
    fn test_vec_sub() {
        let a = Vector::new_from_data([1.0, 2.0, 3.0]);
        let b = Vector::new_from_data([4.0, 5.0, 6.0]);
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
    fn test_vec_mul() {
        let a = Vector::<f64, 3>::new_from_data([1.0, 2.0, 3.0]);
        let b = Vector::<f64, 3>::new_from_data([4.0, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(
            c,
            4.0 + 10.0 + 18.0,
            "Mul did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a,
            b,
            c
        );
    }
    #[test]
    fn test_vec_div() {
        let a = Vector::new_from_data([1.0, 2.0, 3.0]);
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
    fn test_vec_cross_mult() {
        let a: Vector3<f64> = Vector3::<f64>::new_from_data([1.0, 2.0, 3.0]);
        let b: Vector3<f64> = Vector3::<f64>::new_from_data([4.0, 5.0, 6.0]);
        let c = a.cross(b);
        assert_eq!(
            c[0], -3.0,
            "Cross mult did not work. Wrong value at index 0, a:{:?} b:{:?} c:{:?}",
            a, b, c
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
}

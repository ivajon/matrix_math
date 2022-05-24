//! This defines what type of variables are valid for the library, this is used to ensure that the library
//! is not used with any other types.
//! # What is this?
//! This is a trait that is used to ensure that the library is not used with any other types.
//! # Future work
//! This trait is currently in a very early stage of development, I would ideally not use this trait and just set the
//! trait to be all numbers, but I haven't gotten that to work yet
//! # Why is this trait needed?
//! This trait allows the end user to just pass in values of any numerical type, and the library will ensure that
//! the values are valid for the library.
//! 

use num::traits::{Num, NumAssign, NumAssignOps, NumCast, NumOps, One, Zero};
/// Defines a compliant numerical trait
/// It is used as a generic to only allow numbers that can be casted to/ from floats
/// and integers
/// This trait is used to ensure that the matrix struct can only be used with numbers
pub trait CompliantNumerical:
    One + Zero + Num + NumOps + NumAssign + NumAssignOps + Sized + Copy + Clone + NumCast + Default
{
}
impl CompliantNumerical for f32 {}
impl CompliantNumerical for f64 {}
impl CompliantNumerical for i8 {}
impl CompliantNumerical for i16 {}
impl CompliantNumerical for i32 {}
impl CompliantNumerical for i64 {}
impl CompliantNumerical for u8 {}
impl CompliantNumerical for u16 {}
impl CompliantNumerical for u32 {}
impl CompliantNumerical for u64 {}
impl CompliantNumerical for usize {}

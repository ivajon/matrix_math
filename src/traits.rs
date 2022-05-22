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

use std::cmp::PartialOrd;
use std::ops::{Add, Div, Mul, Sub};
/// Defines a compliant numerical trait
/// It is used as a generic to only allow numbers that can be casted to/ from floats
/// and integers
/// This trait is used to ensure that the matrix struct can only be used with numbers
pub trait CompliantNumerical:
    Sized
    + Default
    + Clone
    + Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn into_f64(self) -> f64;
    fn into_f32(self) -> f32;
    fn into_i8(self) -> i8;
    fn into_i16(self) -> i16;
    fn into_i32(self) -> i32;
    fn into_i64(self) -> i64;
    fn into_u8(self) -> u8;
    fn into_u16(self) -> u16;
    fn into_u32(self) -> u32;
    fn into_u64(self) -> u64;
}
impl CompliantNumerical for f64 {
    fn into_f64(self) -> f64 {
        self
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for i32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for i64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for u8 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for u16 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for u32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }

    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self
    }
    fn into_u64(self) -> u64 {
        self as u64
    }
}
impl CompliantNumerical for u64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn into_f32(self) -> f32 {
        self as f32
    }
    fn into_i8(self) -> i8 {
        self as i8
    }
    fn into_i16(self) -> i16 {
        self as i16
    }
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_i64(self) -> i64 {
        self as i64
    }
    fn into_u8(self) -> u8 {
        self as u8
    }
    fn into_u16(self) -> u16 {
        self as u16
    }
    fn into_u32(self) -> u32 {
        self as u32
    }
    fn into_u64(self) -> u64 {
        self
    }
}

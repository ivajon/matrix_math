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
    fn from_f64(f: f64) -> Self;
    fn from_f32(f: f32) -> Self;
    fn from_i8(i: i8) -> Self;
    fn from_i16(i: i16) -> Self;
    fn from_i32(i: i32) -> Self;
    fn from_i64(i: i64) -> Self;
    fn from_u8(i: u8) -> Self;
    fn from_u16(i: u16) -> Self;
    fn from_u32(i: u32) -> Self;
    fn from_u64(i: u64) -> Self;
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
    fn from_f64(f: f64) -> Self {
        f
    }
    fn from_f32(f: f32) -> Self {
        f as f64
    }
    fn from_i8(i: i8) -> Self {
        i as f64
    }
    fn from_i16(i: i16) -> Self {
        i as f64
    }
    fn from_i32(i: i32) -> Self {
        i as f64
    }
    fn from_i64(i: i64) -> Self {
        i as f64
    }
    fn from_u8(i: u8) -> Self {
        i as f64
    }
    fn from_u16(i: u16) -> Self {
        i as f64
    }
    fn from_u32(i: u32) -> Self {
        i as f64
    }
    fn from_u64(i: u64) -> Self {
        i as f64
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
    fn from_f64(f: f64) -> Self {
        f as f32
    }
    fn from_f32(f: f32) -> Self {
        f
    }
    fn from_i8(i: i8) -> Self {
        i as f32
    }
    fn from_i16(i: i16) -> Self {
        i as f32
    }
    fn from_i32(i: i32) -> Self {
        i as f32
    }
    fn from_i64(i: i64) -> Self {
        i as f32
    }
    fn from_u8(i: u8) -> Self {
        i as f32
    }
    fn from_u16(i: u16) -> Self {
        i as f32
    }
    fn from_u32(i: u32) -> Self {
        i as f32
    }
    fn from_u64(i: u64) -> Self {
        i as f32
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
    fn from_f64(f: f64) -> Self {
        f as i32
    }
    fn from_f32(f: f32) -> Self {
        f as i32
    }
    fn from_i8(i: i8) -> Self {
        i as i32
    }
    fn from_i16(i: i16) -> Self {
        i as i32
    }
    fn from_i32(i: i32) -> Self {
        i
    }
    fn from_i64(i: i64) -> Self {
        i as i32
    }
    fn from_u8(i: u8) -> Self {
        i as i32
    }
    fn from_u16(i: u16) -> Self {
        i as i32
    }
    fn from_u32(i: u32) -> Self {
        i as i32
    }
    fn from_u64(i: u64) -> Self {
        i as i32
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
    fn from_f64(f: f64) -> Self {
        f as i64
    }
    fn from_f32(f: f32) -> Self {
        f as i64
    }
    fn from_i8(i: i8) -> Self {
        i as i64
    }
    fn from_i16(i: i16) -> Self {
        i as i64
    }
    fn from_i32(i: i32) -> Self {
        i as i64
    }
    fn from_i64(i: i64) -> Self {
        i
    }
    fn from_u8(i: u8) -> Self {
        i as i64
    }
    fn from_u16(i: u16) -> Self {
        i as i64
    }
    fn from_u32(i: u32) -> Self {
        i as i64
    }
    fn from_u64(i: u64) -> Self {
        i as i64
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
    fn from_f64(f: f64) -> Self {
        f as u8
    }
    fn from_f32(f: f32) -> Self {
        f as u8
    }
    fn from_i8(i: i8) -> Self {
        i as u8
    }
    fn from_i16(i: i16) -> Self {
        i as u8
    }
    fn from_i32(i: i32) -> Self {
        i as u8
    }
    fn from_i64(i: i64) -> Self {
        i as u8
    }
    fn from_u8(i: u8) -> Self {
        i
    }
    fn from_u16(i: u16) -> Self {
        i as u8
    }
    fn from_u32(i: u32) -> Self {
        i as u8
    }
    fn from_u64(i: u64) -> Self {
        i as u8
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
    fn from_f64(f: f64) -> Self {
        f as u16
    }
    fn from_f32(f: f32) -> Self {
        f as u16
    }
    fn from_i8(i: i8) -> Self {
        i as u16
    }
    fn from_i16(i: i16) -> Self {
        i as u16
    }
    fn from_i32(i: i32) -> Self {
        i as u16
    }
    fn from_i64(i: i64) -> Self {
        i as u16
    }
    fn from_u8(i: u8) -> Self {
        i as u16
    }
    fn from_u16(i: u16) -> Self {
        i
    }
    fn from_u32(i: u32) -> Self {
        i as u16
    }
    fn from_u64(i: u64) -> Self {
        i as u16
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
    fn from_f64(f: f64) -> Self {
        f as u32
    }
    fn from_f32(f: f32) -> Self {
        f as u32
    }
    fn from_i8(i: i8) -> Self {
        i as u32
    }
    fn from_i16(i: i16) -> Self {
        i as u32
    }
    fn from_i32(i: i32) -> Self {
        i as u32
    }
    fn from_i64(i: i64) -> Self {
        i as u32
    }
    fn from_u8(i: u8) -> Self {
        i as u32
    }
    fn from_u16(i: u16) -> Self {
        i as u32
    }
    fn from_u64(i: u64) -> Self {
        i as u32
    }

    fn from_u32(i: u32) -> Self {
        i
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
    fn from_f64(f: f64) -> Self {
        f as u64
    }
    fn from_f32(f: f32) -> Self {
        f as u64
    }
    fn from_i8(i: i8) -> Self {
        i as u64
    }
    fn from_i16(i: i16) -> Self {
        i as u64
    }
    fn from_i32(i: i32) -> Self {
        i as u64
    }
    fn from_i64(i: i64) -> Self {
        i as u64
    }
    fn from_u8(i: u8) -> Self {
        i as u64
    }
    fn from_u16(i: u16) -> Self {
        i as u64
    }
    fn from_u32(i: u32) -> Self {
        i as u64
    }

    fn from_u64(i: u64) -> Self {
        i
    }
    
}

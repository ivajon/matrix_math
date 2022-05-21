
use std::cmp::PartialOrd;
use std::ops::{Add, Div, Mul, Sub};
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

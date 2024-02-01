use core::fmt::Display;

use super::CompliantNumerical;
use super::Matrix;

pub mod rotations {
    use crate::CompliantNumerical;

    use super::Matrix;
    #[allow(unused_imports)]
    use num_traits::real::Real;
    #[derive(Debug)]
    pub enum Error {
        InvalidAngle(f32),
    }
    pub trait Trig: CompliantNumerical {
        fn cosine(self) -> Self;
        fn sine(self) -> Self;
        fn pi() -> Self;
    }
    impl Trig for f32 {
        fn cosine(self) -> Self {
            self.cos()
        }
        fn sine(self) -> Self {
            self.sin()
        }
        fn pi() -> Self {
            core::f32::consts::PI
        }
    }
    /// Returns a rotation matrix about the X axis
    ///
    /// Expects radians
    pub fn rotx<T: Trig>(angle: T) -> Result<Matrix<T, 3, 3>, Error> {
        let c = angle.clone().cosine();
        let s = angle.clone().sine();

        let data = [
            [T::one(), T::zero(), T::zero()],
            [T::zero(), c.clone(), T::zero() - s.clone()],
            [T::zero(), s.clone(), c.clone()],
        ];
        Ok(Matrix::from(data))
    }
    /// Returns a rotation matrix about the Y axis
    ///
    /// Expects radians
    pub fn roty<T: Trig>(angle: T) -> Result<Matrix<T, 3, 3>, Error> {
        let c = angle.clone().cosine();
        let s = angle.clone().sine();

        let data = [
            [c.clone(), T::zero(), s.clone()],
            [T::zero(), T::one(), T::zero()],
            [T::zero() - s.clone(), T::zero(), c.clone()],
        ];

        Ok(Matrix::from(data))
    }

    /// Returns a rotation matrix about the Z axis
    ///
    /// Expects radians
    pub fn rotz<T: Trig>(angle: T) -> Result<Matrix<T, 3, 3>, Error> {
        let c = angle.clone().cosine();
        let s = angle.clone().sine();

        let data = [
            [c.clone(), T::zero() - s.clone(), T::zero()],
            [s.clone(), c.clone(), T::zero()],
            [T::zero(), T::zero(), T::one()],
        ];

        Ok(Matrix::from(data))
    }
}

impl<T: CompliantNumerical + Display, const ROWS: usize, const COLS: usize> Display
    for Matrix<T, ROWS, COLS>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for i in 0..ROWS {
            if i == 0 {
                write!(f, "┌ ")?;
            } else if i < ROWS - 1 {
                write!(f, "│ ")?;
            } else {
                write!(f, "└ ")?;
            }

            for j in 0..COLS {
                write!(f, "{}", self[(i, j)])?;

                if j < COLS - 1 {
                    write!(f, " ")?;
                }
            }
            if i == 0 {
                writeln!(f, " ┐")?;
            } else if i == ROWS - 1 {
                writeln!(f, " ┘")?;
            } else {
                writeln!(f, " │")?;
            }
        }

        write!(f, " ")?;

        Ok(())
    }
}

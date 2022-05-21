pub mod vec;
pub mod matrix;
// Defines a method to transform a vector with a matrix
pub impl<T: VecElement, const ROWS: usize, const COLS: usize> ops::Mul<Matrix<T, ROWS, COLS>>
    for Vec<T, ROWS>
{
    type Output = Vec<T, ROWS>;
    fn mul(self, other: Matrix<T, ROWS, COLS>) -> Vec<T, ROWS> {
        let mut result = Vec::new();
        for row in 0..ROWS {
            let mut sum = T::default();
            for col in 0..COLS {
                sum += *self.get(row) * *other.get(row, col);
            }
            result.set(row, sum);
        }
        result
    }
}
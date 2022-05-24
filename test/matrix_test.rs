
#[cfg(test)]
mod tests {
    use crate::matrix::*;
    #[test]
    fn test_matrix_new() {
        let m = Matrix::<u32, 1, 1>::new();
        assert_eq!(m.get(0, 0), &0);
    }
    #[test]
    fn test_matrix_new_from_data() {
        let data = [[1, 2, 3]; 3];
        let m = Matrix::new_from_data(data);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(0, 1), &2);
        assert_eq!(m.get(0, 2), &3);
        assert_eq!(m.get(1, 0), &1);
        assert_eq!(m.get(1, 1), &2);
        assert_eq!(m.get(1, 2), &3);
        assert_eq!(m.get(2, 0), &1);
        assert_eq!(m.get(2, 1), &2);
        assert_eq!(m.get(2, 2), &3);
    }
    #[test]
    fn test_matrix_set() {
        let mut m = Matrix::<u32, 1, 1>::new();
        m.set(0, 0, 1);
        assert_eq!(m.get(0, 0), &1);
    }
    #[test]
    fn test_matrix_add() {
        let mut m1 = Matrix::<u32, 1, 1>::new();
        let mut m2 = Matrix::<u32, 1, 1>::new();
        m1.set(0, 0, 1);
        m2.set(0, 0, 1);
        let m3 = m1 + m2;
        assert_eq!(m3.get(0, 0), &2);
    }
    #[test]
    fn test_matrix_sub() {
        let mut m1 = Matrix::<i32, 1, 1>::new();
        let mut m2 = Matrix::<i32, 1, 1>::new();
        m1.set(0, 0, 0);
        m2.set(0, 0, 1);
        let m3 = m1 - m2;
        assert_eq!(m3.get(0, 0), &-1);
    }
    #[test]
    fn test_matrix_mul_scalar() {
        let mut m1 = Matrix::<u32, 1, 1>::new();
        m1.set(0, 0, 1);
        let m2 = m1 * 2;
        assert_eq!(m2.get(0, 0), &2);
    }
    #[test]
    fn test_matrix_transpose() {
        let data = [[1, 2, 3]; 3];
        let mut m = Matrix::new_from_data(data);
        m = m.transpose();

        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &2);
        assert_eq!(m.get(2, 0), &3);
        assert_eq!(m.get(0, 1), &1);
        assert_eq!(m.get(1, 1), &2);
        assert_eq!(m.get(2, 1), &3);
        assert_eq!(m.get(0, 2), &1);
        assert_eq!(m.get(1, 2), &2);
        assert_eq!(m.get(2, 2), &3);
    }
    #[test]
    fn test_matrix_mul_matrix() {
        let data1 = [[1, 2, 3]; 3];
        let m1 = Matrix::new_from_data(data1);
        let data2 = [[1, 2, 3]; 3];
        let m2 = Matrix::new_from_data(data2);
        let m3 = m1 * m2;
        println!("{:?}", m3);
        assert_eq!(m3.get(0, 0), &6);
        assert_eq!(m3.get(0, 1), &12);
        assert_eq!(m3.get(0, 2), &18);
        assert_eq!(m3.get(1, 0), &6);
        assert_eq!(m3.get(1, 1), &12);
        assert_eq!(m3.get(1, 2), &18);
        assert_eq!(m3.get(2, 0), &6);
        assert_eq!(m3.get(2, 1), &12);
        assert_eq!(m3.get(2, 2), &18);
    }
    #[test]
    fn test_matrix_mul_matrix_non_square() {
        let data1: [[u32; 3]; 4] = [[1, 2, 3]; 4];
        let m1 = Matrix::new_from_data(data1).to_f32();
        let data2: [[i32; 4]; 3] = [[1, 2, 3, 4]; 3];
        let m2 = Matrix::new_from_data(data2).to_f32();
        let m3 = m1 * m2;
        assert_eq!(m3.get(0, 0), &6.0);
        assert_eq!(m3.get(0, 1), &12.0);
        assert_eq!(m3.get(0, 2), &18.0);
        assert_eq!(m3.get(0, 3), &24.0);
        assert_eq!(m3.get(1, 0), &6.0);
        assert_eq!(m3.get(1, 1), &12.0);
        assert_eq!(m3.get(1, 2), &18.0);
        assert_eq!(m3.get(1, 3), &24.0);
        assert_eq!(m3.get(2, 0), &6.0);
        assert_eq!(m3.get(2, 1), &12.0);
        assert_eq!(m3.get(2, 2), &18.0);
        assert_eq!(m3.get(2, 3), &24.0);
    }
}

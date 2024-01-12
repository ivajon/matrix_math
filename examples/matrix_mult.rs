use matrs::predule::*;
fn main() {
    let m1: Matrix<u32, 1, 2> = [[0, 2]].into();
    let m2: Matrix<u32, 2, 1> = [[1], [2]].into();
    println!("{:?}", m1 * m2);
}

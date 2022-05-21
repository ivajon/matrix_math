mod vec;
pub mod matrix;

fn main() {
    let mut v = vec::Vec::<u32,10>::new();
    let mut u = vec::Vec::<u32,10>::new();
    u.set(1,1);
    v.set(0,1);
    let x = v+u;
    println!("v = {:?},u = {:?},x = {:?}",v,u,x);
    println!("Hello, world!");
}

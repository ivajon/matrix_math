# matrix_math

[![build](https://github.com/ivario123/matrix_math/actions/workflows/build.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/build.yml)
[![run_tests](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml)

## What is this?

This is my matrix math library, it is at worst linear memory and hopefully constant memory for everything that does not need instantion of temporary variables. It is also written to allocate memory at compile time and not use dynamic memory allocation. This is done to optimize it for embedded systems and neural networks where we know the size of each layer, it is not good for neat tho, that would require a lot of re allocating memory.

## How it works

Matricies and vectors support the basic linear algebra operations.
Read through the [docs](https://ivario123.github.io/matrix_math/) for more information. Instantiating matricies and vectors is simple and is done with the `new` keyword.

```rust
    use matrix_math::*;

    // Instantiating without initial values
    let mut M = Matrix::<u32, 2, 2>::new();
    let mut V = Vec::<u32, 2>::new();

    // Instantiating with a predefined value
    let mut Vals: [i32; 4] = [1, 2, 3, 4];
    let mut V = Vec::new_from_data(Vals);

    // Instantianting with initial values
    let mut M = Matrix::<u32, 2, 2>::new_from_data([1, 2, 3, 4]);
    let mut V: Vec<u32, 2> = Vec::new_from_data([1, 2]);

    // Using predefined initializers
    let mut M: Matrix<u32, 2, 2> = Matrix::identity();
    let mut V = Vec3::<f32>::new();

    // Using array like indexing to modify values
    M[(0, 0)] = 5;
    V[0] = 5.0;

    // Using function api
    M.set(0, 0, 5);
    V.set(0, 5.0);

    // Using function api to get values
    assert_eq!(*M.get(0, 0), 5);
    assert_eq!(*V.get(0), 5.0);

    // Using array like indexing to get values
    assert_eq!(M[(0, 0)], 5);
    assert_eq!(V[0], 5.0);

    // Using function api to get the size
    assert_eq!(M.size(), (2, 2));
    assert_eq!(V.len(), 2);


```

Every type in this library is generic, so a matrix/vector can contain anny data supporting the basic operations.

Since the matricies are statically allocated there are some things that are hard to do/ can't be done, like computing determinants using the standard algorigthms

## How to use it

See the examples in the [docs](https://ivario123.github.io/matrix_math/)

## Future plans

- Implement usage of tpu and gpus for faster computation.
- Implement usage of multiple threads for faster computation.
- Implement more optimized algorithms.
- Implement for specific compile targets, such as RISC-v vector extensions.
- Implement more mathematical operations
- Implement more rigorous error handling.

## License

This software is provided with zero warranty.
And it is free to use. For more info please see the [license](/LICENSE.md)

## Contributing

Before contributing please read the entire [docs](https://ivario123.github.io/matrix_math/) to make sure you understand what the goal of the project is.
Also read the [contributing guide](.github/CONTRIBUTING.md), then if you have any questions please ask on the [github issues](github.com/ivario123/matrix_math/issues). Finally when you have anny code to submit feel free to fork the repo and submit a pull request.

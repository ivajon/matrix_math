# matrix_math

[![build](https://github.com/ivario123/matrix_math/actions/workflows/build.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/build.yml)
[![run_tests](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml)

## What is this?

This is my matrix math library, it is at worst linear memory and hopefully constant memory for everything that does not need instantiation of temporary variables. It is also written to allocate memory at compile time and not use dynamic memory allocation. This is done to optimize it for embedded systems and neural networks where we know the size of each layer, it is not good for neat tho, as that would require a lot of reallocating memory.

# So you want to use this in your project?

Just add this line to your cargo.toml 

```Toml
matrs = "0.2.0"
```

## How it works

Matrices and vectors support the basic linear algebra operations.
Read through the [docs](https://ivario123.github.io/matrix_math/) for more information.

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
Also, read the [contributing guide ( Not done yet )](.github/CONTRIBUTING.md), then if you have any questions please ask on the [github issues]([github.com/ivario123/matrix_math/issues](https://github.com/ivario123/matrix_math/issues)). Finally when you have any code to submit feel free to fork the repo and submit a pull request.

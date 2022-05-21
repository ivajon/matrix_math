# matrix_math

[![build](https://github.com/ivario123/matrix_math/actions/workflows/build.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/build.yml)
[![run_tests](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml/badge.svg)](https://github.com/ivario123/matrix_math/actions/workflows/run_tests.yml)

## What is this?

This is my matrix math library, it is at worst linear memory and hopefully constant memory for everything that does not need instantion of temporary variables. It is also written to allocate memory at compile time and not use dynamic memory allocation. This is done to optimize it for embedded systems and neural networks where we know the size of each layer, it is not good for neat tho, that would require a lot of re allocating memory.

I intend to use it for some hobby projects, mostly for neural networks.

## How it works

Every type in this library is generic, so a matrix/vector can contain anny data supporting the basic
operations.

## How to use it

See the examples in the `examples` folder. ( Future )

See the tests in the src files.

## Future plans

- Implement usage of tpu and gpus for faster computation.
- Implement usage of multiple threads for faster computation.
- Implement more optimized algorithms.
- Implement for specific compile targets, such as RISC-v vector extensions.
- Implement more mathematical operations
- Implement more rigorous error handling.
- Implement more rigorous tests.

## License

This software is provided with zero warranty.
And it is free to use.

## Contributing

To contribute to this project, please fork it on github and send a pull request.
All contributions need to follow the style guide lines in the `.github/CONTRIBUTING.md` file.

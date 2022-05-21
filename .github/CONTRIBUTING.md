# Style guide

## Follow the rust style guide

All function names and variable names should be per Rust convention.

## File naming

Files should be named in such a way that it's trivial to know what they contain.
File names should only be a few words at most.

## Documentation comments

All files should have a short summary of what they contain. 

```rust
//! # Summary
//! 
```

notation

All functions should have a short summary of what they do. Using

```rust
/// # Summary
/// 
```

notation

All structs should have a short summary of what they contain. Using

```rust
/// # Summary
/// 
```

notation

All enums should have a short summary of what they contain. Using

```rust
/// # Summary
/// 
```

notation

Major sections of code need comments that describe what they contain.
To make sure that users can understand the code.

## Code style

Indentation should be 4 spaces. ( Just set tab to 4 space in your editor of choise )

```rust
fn main() {
    let x = 1;
    let y = 2;
    let z = x + y;
}
```

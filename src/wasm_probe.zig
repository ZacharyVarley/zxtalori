//! Compiler/loader check only. The product's engine ABI comes in implementation.
export fn setup_probe(value: u32) u32 {
    return value *% 2 +% 1;
}

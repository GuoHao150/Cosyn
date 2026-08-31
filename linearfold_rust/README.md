# linearfold_rust

A pure-Rust implementation of the LinearFold algorithm used by Cosyn for MFE (Minimum Free Energy) and RNA secondary structure prediction.


## Integration

`src/mfe_binding/call_mfe.rs` in the main crate dispatches MFE requests to `linearfold_rust::rna_linear_mfe()`. The C++ LinearFold backend has been removed; this Rust crate is now the only MFE backend.

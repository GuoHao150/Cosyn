//! Pure-Rust LinearFold implementation.
//!
//! This crate mirrors the C++ LinearFold reference used by `cosyn` for MFE
//! prediction. It is intentionally kept close to the original algorithm so that
//! outputs can be validated against the C++ binary.

pub mod backtrace;
pub mod constants;
pub mod eval;
pub mod params;
pub mod parser;
pub mod pruning;
pub mod scoring;
pub mod state;

/// Compute MFE and secondary structure for an RNA sequence.
///
/// Mirrors the C++ `rna_linear_mfe()` entry point: runs the beam-CKY parser,
/// then verifies the resulting structure with `eval()` to obtain the final MFE.
pub fn rna_linear_mfe(seq: &str) -> (f64, String) {
    rna_linear_mfe_with_options(seq, 100, true, None)
}

/// Compute MFE with optional maximum base-pair span.
///
/// `max_pair_dist` limits base pairs to span at most this many nucleotides.
/// Setting it to `None` disables the limit (full global folding).
pub fn rna_linear_mfe_with_options(
    seq: &str,
    beam: usize,
    no_sharp_turn: bool,
    max_pair_dist: Option<usize>,
) -> (f64, String) {
    let mut parser = parser::LinearFold::with_max_pair_dist(beam, no_sharp_turn, max_pair_dist);
    let result = parser.parse(seq);
    let energy = eval::eval(seq, &result.structure, false, 2);
    (energy as f64 / -100.0, result.structure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_hairpin() {
        let (mfe, structure) = rna_linear_mfe("GGGAAACCC");
        assert!((mfe - -1.20).abs() < 0.001);
        assert_eq!(structure, "(((...)))");
    }

    #[test]
    fn test_two_hairpins() {
        let (mfe, structure) = rna_linear_mfe("GGGAAACCCGGGAAACCC");
        assert!((mfe - -4.40).abs() < 0.001);
        assert_eq!(structure, "(((...)))(((...)))");
    }

    #[test]
    fn test_120nt() {
        let seq = "AUGCCUAGAAAUUAUUUUCUAGGCAUUUUUUCUUUACAAAAAAAUAAAAGUGUUGUACACUGCUCAGUAGAAAUCCGCCACAAGGGCUACAGGAGCAGUGUCAUGGUCAGCGAUAGCACU";
        let (mfe, structure) = rna_linear_mfe(seq);
        assert!((mfe - -41.00).abs() < 0.001);
        assert_eq!(
            structure,
            "(((((((((((....)))))))))))......................(((((((((((((((((.((((...(((.......)))))))..)))))))))...........))))))))"
        );
    }
}

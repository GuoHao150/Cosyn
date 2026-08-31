//! Shared constants and nucleotide encoding for the Vienna integer model.

/// Number of nucleotide types (N, A, C, G, U).
pub const NOTON: usize = 5;

/// Maximum length of a single-stranded segment in a single-branch loop.
pub const SINGLE_MAX_LEN: usize = 30;

/// Maximum size of a hairpin loop.
pub const HAIRPIN_MAX_LEN: usize = 30;

/// Maximum size of a bulge loop.
pub const BULGE_MAX_LEN: usize = SINGLE_MAX_LEN;

/// Maximum total loop size for interior loops.
pub const MAXLOOP: usize = 30;

/// Minimum beam size to enable cube pruning for M2 = M + P.
pub const MIN_CUBE_PRUNING_SIZE: usize = 20;

/// Map an RNA base to its Vienna encoding: N=0, A=1, C=2, G=3, U=4.
pub fn encode_base(c: char) -> usize {
    match c {
        'A' => 1,
        'C' => 2,
        'G' => 3,
        'U' => 4,
        _ => 0,
    }
}

/// Allowed canonical base pairs (including GU wobble) under Vienna encoding.
pub const ALLOWED_PAIRS: [[bool; NOTON]; NOTON] = {
    let mut arr = [[false; NOTON]; NOTON];
    arr[1][4] = true; // A-U
    arr[4][1] = true; // U-A
    arr[2][3] = true; // C-G
    arr[3][2] = true; // G-C
    arr[3][4] = true; // G-U
    arr[4][3] = true; // U-G
    arr
};

/// Return whether two encoded nucleotides can pair.
pub fn is_pair(nuci: usize, nucj: usize) -> bool {
    ALLOWED_PAIRS[nuci][nucj]
}

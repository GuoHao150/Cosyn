//! Vienna thermodynamic scoring functions for RNA secondary structure elements.
//!
//! All energies are returned in centi-kcal/mol (integer tenths of kcal/mol),
//! matching the C++ `value_type = int` convention.

use crate::constants::*;
use crate::params::*;

/// Map two encoded nucleotides to a Vienna pair type.
///
/// Pair types: 0=NP, 1=CG, 2=GC, 3=GU, 4=UG, 5=AU, 6=UA, 7=NN.
pub fn pair_type(x: usize, y: usize) -> usize {
    match (x, y) {
        (1, 4) => 5, // A-U
        (2, 3) => 1, // C-G
        (3, 2) => 2, // G-C
        (3, 4) => 3, // G-U
        (4, 3) => 4, // U-G
        (4, 1) => 6, // U-A
        _ => 0,
    }
}

/// Scan the sequence for special hairpin loops (tetra-, tri-, and hexaloops).
pub fn init_special_loops(
    seq: &str,
    if_tetraloops: &mut Vec<i32>,
    if_hexaloops: &mut Vec<i32>,
    if_triloops: &mut Vec<i32>,
) {
    let seq_length = seq.len();
    let bytes = seq.as_bytes();

    // Tetraloops: positions i..i+5 with closing C-G.
    if_tetraloops.resize(seq_length.saturating_sub(5), -1);
    for i in 0..seq_length.saturating_sub(5) {
        if bytes[i] != b'C' || bytes[i + 5] != b'G' {
            continue;
        }
        let sub = &seq[i..i + 6];
        if let Some(pos) = Tetraloops.iter().position(|&t| t == sub) {
            if_tetraloops[i] = pos as i32;
        }
    }

    // Triloops: positions i..i+4 with closing C-G or G-C.
    if_triloops.resize(seq_length.saturating_sub(4), -1);
    for i in 0..seq_length.saturating_sub(4) {
        let closing_ok = (bytes[i] == b'C' && bytes[i + 4] == b'G')
            || (bytes[i] == b'G' && bytes[i + 4] == b'C');
        if !closing_ok {
            continue;
        }
        let sub = &seq[i..i + 5];
        if let Some(pos) = Triloops.iter().position(|&t| t == sub) {
            if_triloops[i] = pos as i32;
        }
    }

    // Hexaloops: positions i..i+7 with closing A-U.
    if_hexaloops.resize(seq_length.saturating_sub(7), -1);
    for i in 0..seq_length.saturating_sub(7) {
        if bytes[i] != b'A' || bytes[i + 7] != b'U' {
            continue;
        }
        let sub = &seq[i..i + 8];
        if let Some(pos) = Hexaloops.iter().position(|&t| t == sub) {
            if_hexaloops[i] = pos as i32;
        }
    }
}

/// Score a hairpin loop closed by pair `(i, j)`.
pub fn score_hairpin(
    i: usize,
    j: usize,
    nuci: usize,
    nuci1: usize,
    nucj_1: usize,
    nucj: usize,
    special_index: i32,
) -> i32 {
    let size = j - i - 1;
    let ptype = pair_type(nuci, nucj);

    let mut energy = if size <= HAIRPIN_MAX_LEN {
        hairpin37[size]
    } else {
        hairpin37[30] + (LXC37 * ((size as f64) / 30.0).ln()) as i32
    };

    if size < 3 {
        return energy;
    }

    if size == 4 && special_index > -1 {
        return Tetraloop37[special_index as usize];
    } else if size == 6 && special_index > -1 {
        return Hexaloop37[special_index as usize];
    } else if size == 3 {
        if special_index > -1 {
            return Triloop37[special_index as usize];
        }
        return energy + if ptype > 2 { TERMINAL_AU37 } else { 0 };
    }

    energy += mismatchH37[ptype][nuci1][nucj_1];
    energy
}

/// Score a single-branch loop (stack, bulge, or interior loop).
#[allow(clippy::too_many_arguments)]
pub fn score_single(
    i: usize,
    j: usize,
    p: usize,
    q: usize,
    nuci: usize,
    nuci1: usize,
    nucj_1: usize,
    nucj: usize,
    nucp_1: usize,
    nucp: usize,
    nucq: usize,
    nucq1: usize,
) -> i32 {
    let ptype = pair_type(nuci, nucj);
    let ptype2 = pair_type(nucq, nucp);
    let n1 = p - i - 1;
    let n2 = j - q - 1;

    let (nl, ns) = if n1 > n2 { (n1, n2) } else { (n2, n1) };
    let mut energy;

    if nl == 0 {
        return stack37[ptype][ptype2];
    }

    if ns == 0 {
        // Bulge loop.
        energy = if nl <= MAXLOOP {
            bulge37[nl]
        } else {
            bulge37[30] + (LXC37 * ((nl as f64) / 30.0).ln()) as i32
        };
        if nl == 1 {
            energy += stack37[ptype][ptype2];
        } else {
            if ptype > 2 {
                energy += TERMINAL_AU37;
            }
            if ptype2 > 2 {
                energy += TERMINAL_AU37;
            }
        }
        return energy;
    }

    // Interior loop.
    if ns == 1 {
        if nl == 1 {
            // 1x1 interior loop.
            return int11_37[ptype][ptype2][nuci1][nucj_1];
        }
        if nl == 2 {
            // 2x1 interior loop.
            return if n1 == 1 {
                int21_37[ptype][ptype2][nuci1][nucq1][nucj_1]
            } else {
                int21_37[ptype2][ptype][nucq1][nuci1][nucp_1]
            };
        }
        // 1xn interior loop.
        let sz = nl + 1;
        energy = if sz <= MAXLOOP {
            internal_loop37[sz]
        } else {
            internal_loop37[30] + (LXC37 * ((sz as f64) / 30.0).ln()) as i32
        };
        energy += std::cmp::min(MAX_NINIO, ((nl - ns) as i32) * NINIO37);
        energy += mismatch1nI37[ptype][nuci1][nucj_1] + mismatch1nI37[ptype2][nucq1][nucp_1];
        return energy;
    }

    if ns == 2 {
        if nl == 2 {
            // 2x2 interior loop.
            return int22_37[ptype][ptype2][nuci1][nucp_1][nucq1][nucj_1];
        } else if nl == 3 {
            // 2x3 interior loop.
            energy = internal_loop37[5] + NINIO37;
            energy += mismatch23I37[ptype][nuci1][nucj_1] + mismatch23I37[ptype2][nucq1][nucp_1];
            return energy;
        }
    }

    // Generic interior loop.
    let u = nl + ns;
    energy = if u <= MAXLOOP {
        internal_loop37[u]
    } else {
        internal_loop37[30] + (LXC37 * ((u as f64) / 30.0).ln()) as i32
    };
    energy += std::cmp::min(MAX_NINIO, ((nl - ns) as i32) * NINIO37);
    energy += mismatchI37[ptype][nuci1][nucj_1] + mismatchI37[ptype2][nucq1][nucp_1];
    energy
}

/// Score a multi-loop stem (M1 or closing pair).
fn multi_loop_stem(ptype: usize, si1: i32, sj1: i32, dangle_model: i32) -> i32 {
    let mut energy = 0;
    if dangle_model != 0 && si1 >= 0 && sj1 >= 0 {
        energy += mismatchM37[ptype][si1 as usize][sj1 as usize];
    }
    if ptype > 2 {
        energy += TERMINAL_AU37;
    }
    energy += ML_INTERN37;
    energy
}

/// Score an M1 stem inside a multi-loop.
pub fn score_m1(nuci_1: i32, nuci: usize, nuck: usize, nuck1: i32, dangle_model: i32) -> i32 {
    let tt = pair_type(nuci, nuck);
    multi_loop_stem(tt, nuci_1, nuck1, dangle_model)
}

/// Score an unpaired region inside a multi-loop (current model returns 0).
pub fn score_multi_unpaired() -> i32 {
    0
}

/// Score the closing pair of a multi-loop.
pub fn score_multi(
    nuci: usize,
    nuci1: usize,
    nucj_1: usize,
    nucj: usize,
    dangle_model: i32,
) -> i32 {
    let tt = pair_type(nucj, nuci);
    multi_loop_stem(tt, nucj_1 as i32, nuci1 as i32, dangle_model) + ML_CLOSING37
}

/// Score a base pair in the external loop.
pub fn score_external_paired(
    nuci_1: i32,
    nuci: usize,
    nucj: usize,
    nucj1: i32,
    dangle_model: i32,
) -> i32 {
    let ptype = pair_type(nuci, nucj);
    let mut energy = 0;
    if dangle_model != 0 {
        if nuci_1 >= 0 && nucj1 >= 0 {
            energy += mismatchExt37[ptype][nuci_1 as usize][nucj1 as usize];
        } else if nuci_1 >= 0 {
            energy += dangle5_37[ptype][nuci_1 as usize];
        } else if nucj1 >= 0 {
            energy += dangle3_37[ptype][nucj1 as usize];
        }
    }
    if ptype > 2 {
        energy += TERMINAL_AU37;
    }
    energy
}

/// Score an unpaired base in the external loop (current model returns 0).
pub fn score_external_unpaired() -> i32 {
    0
}

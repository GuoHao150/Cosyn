//! Evaluate the free energy of a given RNA secondary structure.
//!
//! This walks the dot-bracket string with an explicit stack and sums the same
//! scoring terms used by the DP parser.

use crate::constants::*;
use crate::scoring::*;

pub fn eval(seq: &str, structure: &str, is_verbose: bool, dangle_model: i32) -> i64 {
    let seq_length = seq.len();
    let seq_bytes = seq.as_bytes();

    let mut if_tetraloops: Vec<i32> = Vec::new();
    let mut if_hexaloops: Vec<i32> = Vec::new();
    let mut if_triloops: Vec<i32> = Vec::new();

    init_special_loops(seq, &mut if_tetraloops, &mut if_hexaloops, &mut if_triloops);

    let eval_nucs: Vec<usize> = seq.bytes().map(|b| encode_base(b as char)).collect();

    let mut total_energy: i64 = 0;
    let mut external_energy: i64 = 0;
    let mut m1_energy: Vec<i64> = vec![0; seq_length];
    let mut multi_number_unpaired: Vec<i64> = vec![0; seq_length];

    // Stack of (index, page)
    let mut stk: Vec<(usize, usize)> = Vec::new();
    let mut inner_loop: Option<(usize, usize)> = None;

    for j in 0..seq_length {
        m1_energy[j] = 0;
        multi_number_unpaired[j] = 0;

        let c = structure.as_bytes()[j] as char;

        if c == '.' {
            if let Some(&(top_i, _)) = stk.last() {
                multi_number_unpaired[top_i] += 1;
            }
        } else if c == '(' {
            if let Some(top) = stk.last_mut() {
                top.1 += 1;
            }
            stk.push((j, 0));
        } else if c == ')' {
            assert!(!stk.is_empty());
            let (i, page) = stk.pop().unwrap();

            let nuci = eval_nucs[i];
            let nucj = eval_nucs[j];
            let nuci1 = if i + 1 < seq_length {
                eval_nucs[i + 1] as i32
            } else {
                -1
            };
            let nucj_1 = if j > 0 { eval_nucs[j - 1] as i32 } else { -1 };
            let nuci_1 = if i > 0 { eval_nucs[i - 1] as i32 } else { -1 };
            let nucj1 = if j + 1 < seq_length {
                eval_nucs[j + 1] as i32
            } else {
                -1
            };

            if page == 0 {
                // Hairpin.
                let mut special: i32 = -1;
                let loop_size = j - i - 1;
                if loop_size == 4 {
                    special = if_tetraloops[i];
                } else if loop_size == 6 {
                    special = if_hexaloops[i];
                } else if loop_size == 3 {
                    special = if_triloops[i];
                }
                let newscore =
                    -(score_hairpin(i, j, nuci, nuci1 as usize, nucj_1 as usize, nucj, special)
                        as i64);
                if is_verbose {
                    eprintln!(
                        "Hairpin loop ( {}, {}) {}{} : {:.2}",
                        i + 1,
                        j + 1,
                        seq_bytes[i] as char,
                        seq_bytes[j] as char,
                        newscore as f64 / -100.0
                    );
                }
                total_energy += newscore;
            } else if page == 1 {
                // Interior loop / stack.
                let (p, q) = inner_loop.unwrap();
                let nucp_1 = eval_nucs[p - 1];
                let nucp = eval_nucs[p];
                let nucq = eval_nucs[q];
                let nucq1 = eval_nucs[q + 1];
                let newscore = -(score_single(
                    i,
                    j,
                    p,
                    q,
                    nuci,
                    nuci1 as usize,
                    nucj_1 as usize,
                    nucj,
                    nucp_1,
                    nucp,
                    nucq,
                    nucq1,
                ) as i64);
                if is_verbose {
                    eprintln!(
                        "Interior loop ( {}, {}) {}{}; ( {}, {}) {}{} : {:.2}",
                        i + 1,
                        j + 1,
                        seq_bytes[i] as char,
                        seq_bytes[j] as char,
                        p + 1,
                        q + 1,
                        seq_bytes[p] as char,
                        seq_bytes[q] as char,
                        newscore as f64 / -100.0
                    );
                }
                total_energy += newscore;
            } else {
                // Multi-loop.
                let mut multi_score: i64 = 0;
                multi_score += m1_energy[i];
                multi_score +=
                    -(score_multi(nuci, nuci1 as usize, nucj_1 as usize, nucj, dangle_model)
                        as i64);
                multi_score += -(score_multi_unpaired() as i64);
                if is_verbose {
                    eprintln!(
                        "Multi loop ( {}, {}) {}{} : {:.2}",
                        i + 1,
                        j + 1,
                        seq_bytes[i] as char,
                        seq_bytes[j] as char,
                        multi_score as f64 / -100.0
                    );
                }
                total_energy += multi_score;
            }

            // Update inner_loop.
            inner_loop = Some((i, j));

            // Possible M1 contribution to enclosing multi-loop.
            if let Some(&(top_i, _)) = stk.last() {
                m1_energy[top_i] += -(score_m1(nuci_1, nuci, nucj, nucj1, dangle_model) as i64);
            }

            // External loop contribution.
            if stk.is_empty() {
                let k = i as i32 - 1;
                let nuck = if k > -1 {
                    eval_nucs[k as usize] as i32
                } else {
                    -1
                };
                let nuck1 = eval_nucs[(k + 1) as usize];
                external_energy +=
                    -(score_external_paired(nuck, nuck1, nucj, nucj1, dangle_model) as i64);
            }
        }
    }

    if is_verbose {
        eprintln!("External loop : {:.2}", external_energy as f64 / -100.0);
    }
    total_energy += external_energy;
    total_energy
}

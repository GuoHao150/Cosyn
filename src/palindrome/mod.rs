use serde_json::{json, Value};
use std::collections::BTreeSet;

/// Supported separator sizes between palindrome arms.
const SEP_SIZES: [usize; 2] = [4, 5];
/// Minimum bases per palindrome arm to report.
const MIN_PALIN_SIZE: usize = 4;

#[derive(Clone, Copy)]
struct PalindromicSeq<'a> {
    seq: &'a str,
    start_idx: usize,
    end_idx: usize,
}

fn base_matched(n1: &str, n2: &str) -> bool {
    match (n1, n2) {
        ("A", "T") | ("A", "U") | ("T", "A") | ("U", "A") | ("G", "C") | ("C", "G") => true,
        _ => false,
    }
}

/// Single-pass palindrome search over `inputs`.
///
/// For every position *i* we try both separator sizes (4 and 5).  When the
/// innermost pair matches, we expand outward as far as possible, recording
/// every palindrome whose arm length is ≥ `MIN_PALIN_SIZE` (4).  Results are
/// deduplicated by (start, end) via a `BTreeSet`.
///
/// This replaces the previous 18-pass approach (9 palindrome sizes × 2
/// separator sizes) with a single linear scan, reducing the asymptotic
/// work from ≈18n² to ≈2n² with a much smaller constant factor.
fn find_all_palindromes(inputs: &str) -> Vec<PalindromicSeq<'_>> {
    let n = inputs.len();
    if n < MIN_PALIN_SIZE * 2 + SEP_SIZES[0] {
        return vec![];
    }

    let mut results: Vec<PalindromicSeq<'_>> = Vec::new();
    let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();

    // Scan every position as the rightmost base of the left palindrome arm.
    let max_i = n.saturating_sub(SEP_SIZES[0] + 2); // need room for sep + right base
    for i in MIN_PALIN_SIZE - 1..max_i {
        for &sep in &SEP_SIZES {
            let mut left = i;
            let mut right = left + sep + 1;
            if right >= n || !base_matched(&inputs[left..left + 1], &inputs[right..right + 1]) {
                continue;
            }

            // Expand outward.
            loop {
                let can_expand = left > 0
                    && right + 1 < n
                    && base_matched(&inputs[left - 1..left], &inputs[right + 1..right + 2]);
                if can_expand {
                    left -= 1;
                    right += 1;
                } else {
                    break;
                }
            }

            // Record the maximal palindrome and all shorter variants down to
            // MIN_PALIN_SIZE.  For each, check (start, end) uniqueness.
            let max_arm_len = (right - left + 1 - sep) / 2;
            for arm_len in (MIN_PALIN_SIZE..=max_arm_len).rev() {
                let inner_offset = max_arm_len - arm_len;
                let start = left + inner_offset;
                let end = right - inner_offset + 1;
                if seen.insert((start, end)) {
                    results.push(PalindromicSeq {
                        seq: &inputs[start..end],
                        start_idx: start,
                        end_idx: end,
                    });
                }
            }
        }
    }
    results
}

/// Calculate the palindromic score.
///
/// Returns `(score, palindrome_sequences)` where `score = count + coverage`
/// and `coverage` is the fraction of bases covered by any palindrome,
/// scaled by sequence length.
pub(crate) fn palindrome_score(inputs: &str) -> (f64, Vec<String>) {
    let n = inputs.len();
    if n == 0 {
        return (0.0, vec![]);
    }

    let palin_results = find_all_palindromes(inputs);
    let palindrome_nums = palin_results.len();

    // Coverage: track which positions fall inside any palindrome.
    // Vec<bool> is faster than BTreeSet for dense position sets.
    let mut covered = vec![false; n];
    let mut palin_seqs: Vec<String> = Vec::with_capacity(palindrome_nums);
    for p in &palin_results {
        palin_seqs.push(p.seq.to_string());
        for j in p.start_idx..p.end_idx {
            covered[j] = true;
        }
    }

    let covered_count = covered.iter().filter(|&&b| b).count();
    let palindrome_coverage = (covered_count as f64 / n as f64) * n as f64;
    let score = palindrome_nums as f64 + palindrome_coverage;

    if score.is_nan() {
        (0.0, palin_seqs)
    } else {
        (score, palin_seqs)
    }
}

pub(crate) struct PalinCalculator<'a> {
    seq_id: &'a String,
    dna_seq: &'a String,
}

impl<'a> PalinCalculator<'a> {
    pub(crate) fn new(seq_id: &'a String, dna_seq: &'a String) -> Self {
        PalinCalculator { seq_id, dna_seq }
    }

    pub(crate) fn get_palindrome_seqs(&self) -> String {
        let palin_results = find_all_palindromes(self.dna_seq);
        let mut palin_seqs: Vec<String> = vec![];
        for p in palin_results.iter() {
            palin_seqs.push(p.seq.to_string());
        }
        let palin_nums = palin_seqs.len();
        let merged_seqs = palin_seqs.join(";");
        format!("{}-{}", palin_nums, merged_seqs)
    }

    pub(crate) fn to_json(&self) -> Value {
        let palin_out = self.get_palindrome_seqs();
        json!({
            "seq_id": self.seq_id,
            "palindromic_sequences": palin_out,
        })
    }
}

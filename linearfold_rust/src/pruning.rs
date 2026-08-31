//! Beam pruning and candidate sorting utilities.

use rustc_hash::FxHashMap as HashMap;

use crate::state::{Score, State, SCORE_MIN};

/// Sort the keys of a beam in descending order.
pub fn sort_keys(map: &HashMap<i32, State>, keys: &mut Vec<(i32, State)>) {
    keys.clear();
    for (&k, &v) in map.iter() {
        keys.push((k, v));
    }
    keys.sort_by(|a, b| b.0.cmp(&a.0));
}

/// Sort beam candidates by their combined score with the best prefix C.
pub fn sort_m(
    threshold: Score,
    beamstep: &HashMap<i32, State>,
    sorted_step_m: &mut Vec<(Score, i32)>,
    best_c: &[State],
    scores: &[(Score, i32)],
) {
    sorted_step_m.clear();
    if threshold == SCORE_MIN {
        for (&i, state) in beamstep.iter() {
            let k = i - 1;
            let newscore = if k >= 0 && best_c[k as usize].score == SCORE_MIN {
                SCORE_MIN
            } else {
                (if k >= 0 { best_c[k as usize].score } else { 0 }) + state.score
            };
            sorted_step_m.push((newscore, i));
        }
    } else {
        for &(score, i) in scores.iter() {
            if score >= threshold {
                sorted_step_m.push((score, i));
            }
        }
    }
    sorted_step_m.sort_by(|a, b| b.0.cmp(&a.0));
}

fn quickselect_partition(scores: &mut [(Score, i32)], lower: usize, upper: usize) -> usize {
    let pivot = scores[upper].0;
    let mut lower = lower;
    let mut upper = upper;
    while lower < upper {
        while scores[lower].0 < pivot {
            lower += 1;
        }
        while scores[upper].0 > pivot {
            upper -= 1;
        }
        if scores[lower].0 == scores[upper].0 {
            lower += 1;
        } else if lower < upper {
            scores.swap(lower, upper);
        }
    }
    upper
}

fn quickselect(scores: &mut [(Score, i32)], lower: usize, upper: usize, k: usize) -> Score {
    if lower == upper {
        return scores[lower].0;
    }
    let split = quickselect_partition(scores, lower, upper);
    let length = split - lower + 1;
    if length == k {
        scores[split].0
    } else if k < length {
        if split == 0 {
            scores[lower].0
        } else {
            quickselect(scores, lower, split - 1, k)
        }
    } else {
        quickselect(scores, split + 1, upper, k - length)
    }
}

/// Keep only the top `beam` candidates in a beam step.
///
/// The score used for ranking is `bestC[k].score + cand.score`, where `k = i - 1`.
/// Returns the score threshold used, or `SCORE_MIN` if no pruning occurred.
pub fn beam_prune(
    beamstep: &mut HashMap<i32, State>,
    best_c: &[State],
    scores: &mut Vec<(Score, i32)>,
    beam: usize,
) -> Score {
    scores.clear();
    for (&i, state) in beamstep.iter() {
        let k = i - 1;
        let newscore = if k >= 0 && best_c[k as usize].score == SCORE_MIN {
            SCORE_MIN
        } else {
            (if k >= 0 { best_c[k as usize].score } else { 0 }) + state.score
        };
        scores.push((newscore, i));
    }
    let scores_len = scores.len();
    if scores_len <= beam {
        return SCORE_MIN;
    }
    let threshold = quickselect(scores, 0, scores_len - 1, scores_len - beam);
    let to_remove: Vec<i32> = scores
        .iter()
        .filter(|(score, _)| *score < threshold)
        .map(|(_, i)| *i)
        .collect();
    for i in to_remove {
        beamstep.remove(&i);
    }
    threshold
}

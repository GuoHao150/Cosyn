//! Beam-pruned LinearFold DP parser.

use rustc_hash::FxHashMap as HashMap;

use crate::constants::*;
use crate::pruning::*;
use crate::scoring::*;
use crate::state::*;

pub struct LinearFold {
    pub(crate) beam: usize,
    pub(crate) no_sharp_turn: bool,
    pub(crate) dangle_model: i32,
    /// Optional maximum span (in nucleotides) for base pairs. `None` means no limit.
    pub(crate) max_pair_dist: Option<usize>,

    pub(crate) seq_length: usize,

    pub(crate) best_h: Vec<HashMap<i32, State>>,
    pub(crate) best_p: Vec<HashMap<i32, State>>,
    pub(crate) best_m2: Vec<HashMap<i32, State>>,
    pub(crate) best_multi: Vec<HashMap<i32, State>>,
    pub(crate) best_m: Vec<HashMap<i32, State>>,
    pub(crate) best_c: Vec<State>,

    pub(crate) sorted_best_m: Vec<Vec<(Score, i32)>>,
    pub(crate) keys: Vec<(i32, State)>,
    pub(crate) scores: Vec<(Score, i32)>,

    pub(crate) nucs: Vec<usize>,

    pub(crate) if_tetraloops: Vec<i32>,
    pub(crate) if_hexaloops: Vec<i32>,
    pub(crate) if_triloops: Vec<i32>,
}

impl LinearFold {
    pub fn new(beam: usize, no_sharp_turn: bool) -> Self {
        Self::with_max_pair_dist(beam, no_sharp_turn, None)
    }

    pub fn with_max_pair_dist(
        beam: usize,
        no_sharp_turn: bool,
        max_pair_dist: Option<usize>,
    ) -> Self {
        LinearFold {
            beam,
            no_sharp_turn,
            dangle_model: 2,
            max_pair_dist,
            seq_length: 0,
            best_h: Vec::new(),
            best_p: Vec::new(),
            best_m2: Vec::new(),
            best_multi: Vec::new(),
            best_m: Vec::new(),
            best_c: Vec::new(),
            sorted_best_m: Vec::new(),
            keys: Vec::new(),
            scores: Vec::new(),
            nucs: Vec::new(),
            if_tetraloops: Vec::new(),
            if_hexaloops: Vec::new(),
            if_triloops: Vec::new(),
        }
    }

    #[inline]
    fn pair_span_allowed(&self, left: usize, right: usize) -> bool {
        self.max_pair_dist
            .map_or(true, |max| right.saturating_sub(left) <= max)
    }

    fn prepare(&mut self, len: usize) {
        self.seq_length = len;
        self.best_h = vec![HashMap::default(); len];
        self.best_p = vec![HashMap::default(); len];
        self.best_m2 = vec![HashMap::default(); len];
        self.best_m = vec![HashMap::default(); len];
        self.best_c = vec![State::new(); len];
        self.best_multi = vec![HashMap::default(); len];
        self.sorted_best_m = vec![Vec::new(); len];
        self.nucs = vec![0; len];
        self.scores.reserve(len);
    }

    pub fn parse(&mut self, seq: &str) -> DecoderResult {
        self.prepare(seq.len());
        let seq_length = self.seq_length;

        let seq_bytes = seq.as_bytes();
        for (i, &b) in seq_bytes.iter().enumerate() {
            self.nucs[i] = encode_base(b as char);
        }

        init_special_loops(
            seq,
            &mut self.if_tetraloops,
            &mut self.if_hexaloops,
            &mut self.if_triloops,
        );

        let mut next_pair: Vec<Vec<i32>> = vec![vec![-1; seq_length]; NOTON];
        for nuci in 0..NOTON {
            let mut next = -1i32;
            for j in (0..seq_length).rev() {
                next_pair[nuci][j] = next;
                if ALLOWED_PAIRS[nuci][self.nucs[j]] {
                    next = j as i32;
                }
            }
        }

        if seq_length > 0 {
            self.best_c[0].set(-score_external_unpaired(), Manner::CEqCPlusU);
        }
        if seq_length > 1 {
            self.best_c[1].set(-score_external_unpaired(), Manner::CEqCPlusU);
        }

        for j in 0..seq_length {
            let nucj = self.nucs[j];
            let nucj1 = if j + 1 < seq_length {
                self.nucs[j + 1] as i32
            } else {
                -1
            };

            // Beam of H
            {
                let mut beamstep_h = std::mem::take(&mut self.best_h[j]);

                if self.beam > 0 && beamstep_h.len() > self.beam {
                    beam_prune(&mut beamstep_h, &self.best_c, &mut self.scores, self.beam);
                }

                {
                    let mut jnext = next_pair[nucj][j];
                    if self.no_sharp_turn {
                        while jnext != -1 && jnext - (j as i32) < 4 {
                            jnext = next_pair[nucj][jnext as usize];
                        }
                    }

                    if jnext != -1 && self.pair_span_allowed(j, jnext as usize) {
                        let jnext = jnext as usize;
                        let nucjnext = self.nucs[jnext];
                        let nucjnext_1 = if jnext > 0 {
                            self.nucs[jnext - 1] as i32
                        } else {
                            -1
                        };

                        let mut tetra_hex_tri = -1i32;
                        let loop_size = jnext - j - 1;
                        if loop_size == 4 {
                            tetra_hex_tri = self.if_tetraloops[j];
                        } else if loop_size == 6 {
                            tetra_hex_tri = self.if_hexaloops[j];
                        } else if loop_size == 3 {
                            tetra_hex_tri = self.if_triloops[j];
                        }

                        let newscore = -score_hairpin(
                            j,
                            jnext,
                            nucj,
                            nucj1 as usize,
                            nucjnext_1 as usize,
                            nucjnext,
                            tetra_hex_tri,
                        );
                        self.best_h[jnext]
                            .entry(j as i32)
                            .or_default()
                            .update_if_better(newscore, Manner::H);
                    }
                }

                {
                    sort_keys(&beamstep_h, &mut self.keys);
                    let keys: Vec<(i32, State)> = self.keys.clone();
                    for (i, state) in keys.iter() {
                        let i = *i as usize;
                        let nuci = self.nucs[i];
                        let jnext = next_pair[nuci][j];

                        // Generate p(i, j)
                        {
                            self.best_p[j]
                                .entry(i as i32)
                                .or_default()
                                .update_if_better(state.score, Manner::Hairpin);
                        }

                        if jnext != -1 && self.pair_span_allowed(i, jnext as usize) {
                            let jnext = jnext as usize;
                            let nuci1 = if i + 1 < seq_length {
                                self.nucs[i + 1] as i32
                            } else {
                                -1
                            };
                            let nucjnext = self.nucs[jnext];
                            let nucjnext_1 = if jnext > 0 {
                                self.nucs[jnext - 1] as i32
                            } else {
                                -1
                            };

                            let mut tetra_hex_tri = -1i32;
                            let loop_size = jnext - i - 1;
                            if loop_size == 4 {
                                tetra_hex_tri = self.if_tetraloops[i];
                            } else if loop_size == 6 {
                                tetra_hex_tri = self.if_hexaloops[i];
                            } else if loop_size == 3 {
                                tetra_hex_tri = self.if_triloops[i];
                            }

                            let newscore = -score_hairpin(
                                i,
                                jnext,
                                nuci,
                                nuci1 as usize,
                                nucjnext_1 as usize,
                                nucjnext,
                                tetra_hex_tri,
                            );
                            self.best_h[jnext]
                                .entry(i as i32)
                                .or_default()
                                .update_if_better(newscore, Manner::H);
                        }
                    }
                }

                self.best_h[j] = beamstep_h;
            }

            if j == 0 {
                continue;
            }

            // Beam of Multi
            {
                let mut beamstep_multi = std::mem::take(&mut self.best_multi[j]);

                if self.beam > 0 && beamstep_multi.len() > self.beam {
                    beam_prune(
                        &mut beamstep_multi,
                        &self.best_c,
                        &mut self.scores,
                        self.beam,
                    );
                }

                sort_keys(&beamstep_multi, &mut self.keys);
                let keys: Vec<(i32, State)> = self.keys.clone();
                for (i, state) in keys.iter() {
                    let i = *i as usize;
                    let nuci = self.nucs[i];
                    let nuci1 = self.nucs[i + 1];
                    let jnext = next_pair[nuci][j];

                    // Generate P(i, j)
                    {
                        let newscore = state.score
                            - score_multi(nuci, nuci1, self.nucs[j - 1], nucj, self.dangle_model);
                        self.best_p[j]
                            .entry(i as i32)
                            .or_default()
                            .update_if_better(newscore, Manner::PEqMulti);
                    }

                    if jnext != -1 {
                        let jnext = jnext as usize;
                        if let TraceInfo::Paddings { l1, l2 } = state.trace {
                            let new_l1 = l1;
                            let new_l2 = l2 + jnext as i32 - j as i32;
                            let newscore = state.score - score_multi_unpaired();
                            self.best_multi[jnext]
                                .entry(i as i32)
                                .or_default()
                                .update_if_better_paddings(
                                    newscore,
                                    Manner::MultiEqMultiPlusU,
                                    new_l1,
                                    new_l2,
                                );
                        }
                    }
                }

                self.best_multi[j] = beamstep_multi;
            }

            // Beam of P
            {
                let mut beamstep_p = std::mem::take(&mut self.best_p[j]);
                let mut beamstep_c = self.best_c[j];

                if self.beam > 0 && beamstep_p.len() > self.beam {
                    beam_prune(&mut beamstep_p, &self.best_c, &mut self.scores, self.beam);
                }

                let use_cube_pruning =
                    self.beam > MIN_CUBE_PRUNING_SIZE && beamstep_p.len() > MIN_CUBE_PRUNING_SIZE;

                sort_keys(&beamstep_p, &mut self.keys);
                let keys: Vec<(i32, State)> = self.keys.clone();
                for (i, state) in keys.iter() {
                    let i = *i as usize;
                    let nuci = self.nucs[i];
                    let nuci_1 = if i > 0 { self.nucs[i - 1] as i32 } else { -1 };

                    // M = P
                    if i > 0 && j < seq_length - 6 {
                        let newscore =
                            -score_m1(nuci_1, nuci, nucj, nucj1, self.dangle_model) + state.score;
                        self.best_m[j]
                            .entry(i as i32)
                            .or_default()
                            .update_if_better(newscore, Manner::MEqP);
                    }

                    // M2 = M + P
                    if !use_cube_pruning && j < seq_length - 1 {
                        let k = i as i32 - 1;
                        if k > 0 && !self.best_m[k as usize].is_empty() {
                            let m1_score = -score_m1(nuci_1, nuci, nucj, nucj1, self.dangle_model)
                                + state.score;
                            for (&newi, m_state) in self.best_m[k as usize].iter() {
                                let newscore = m1_score + m_state.score;
                                self.best_m2[j]
                                    .entry(newi)
                                    .or_default()
                                    .update_if_better_split(newscore, Manner::M2EqMPlusP, k);
                            }
                        }
                    }

                    // C = C + P
                    {
                        let k = i as i32 - 1;
                        if k >= 0 {
                            let prefix_c = &self.best_c[k as usize];
                            if !matches!(prefix_c.manner, Manner::None) {
                                let nuck = nuci_1;
                                let nuck1 = nuci;
                                let newscore = -score_external_paired(
                                    nuck,
                                    nuck1,
                                    nucj,
                                    nucj1,
                                    self.dangle_model,
                                ) + prefix_c.score
                                    + state.score;
                                beamstep_c.update_if_better_split(newscore, Manner::CEqCPlusP, k);
                            }
                        } else {
                            let newscore = -score_external_paired(
                                -1,
                                self.nucs[0],
                                nucj,
                                nucj1,
                                self.dangle_model,
                            ) + state.score;
                            beamstep_c.update_if_better_split(newscore, Manner::CEqCPlusP, -1);
                        }
                    }

                    // Generate new helix / single branch
                    if i > 0 && j < seq_length - 1 {
                        let start_p = i.saturating_sub(SINGLE_MAX_LEN);
                        for p in (start_p..i).rev() {
                            let nucp = self.nucs[p];
                            let nucp1 = self.nucs[p + 1];
                            let mut q = next_pair[nucp][j];

                            while q != -1
                                && ((i - p) + (q as usize - j) - 2 <= SINGLE_MAX_LEN)
                                && self.pair_span_allowed(p, q as usize)
                            {
                                let qu = q as usize;
                                let nucq = self.nucs[qu];
                                let nucq_1 = self.nucs[qu - 1];

                                if p == i - 1 && qu == j + 1 {
                                    // helix
                                    let newscore = -score_single(
                                        p,
                                        qu,
                                        i,
                                        j,
                                        nucp,
                                        nucp1,
                                        nucq_1,
                                        nucq,
                                        nuci_1 as usize,
                                        nuci,
                                        nucj,
                                        nucj1 as usize,
                                    ) + state.score;
                                    self.best_p[qu]
                                        .entry(p as i32)
                                        .or_default()
                                        .update_if_better(newscore, Manner::Helix);
                                } else {
                                    // single branch
                                    let newscore = -score_single(
                                        p,
                                        qu,
                                        i,
                                        j,
                                        nucp,
                                        nucp1,
                                        nucq_1,
                                        nucq,
                                        nuci_1 as usize,
                                        nuci,
                                        nucj,
                                        nucj1 as usize,
                                    ) + state.score;
                                    self.best_p[qu]
                                        .entry(p as i32)
                                        .or_default()
                                        .update_if_better_paddings(
                                            newscore,
                                            Manner::Single,
                                            (i - p) as i8,
                                            (qu - j) as i32,
                                        );
                                }
                                q = next_pair[nucp][qu];
                            }
                        }
                    }
                }

                // Cube pruning for M2 = M + P
                if use_cube_pruning && j < seq_length - 1 {
                    let mut valid_ps: Vec<i32> = Vec::new();
                    let mut m1_scores: Vec<Score> = Vec::new();

                    sort_keys(&beamstep_p, &mut self.keys);
                    let keys: Vec<(i32, State)> = self.keys.clone();
                    for (i, state) in keys.iter() {
                        let i = *i as usize;
                        let nuci = self.nucs[i];
                        let nuci_1 = if i > 0 { self.nucs[i - 1] as i32 } else { -1 };
                        let k = i as i32 - 1;

                        if k > 0 && !self.best_m[k as usize].is_empty() {
                            let m1_score = -score_m1(nuci_1, nuci, nucj, nucj1, self.dangle_model)
                                + state.score;
                            valid_ps.push(i as i32);
                            m1_scores.push(m1_score);
                        }
                    }

                    // max heap: (heuristic score, (index in valid_ps, index in sorted_best_m))
                    let mut heap: Vec<(Score, usize, usize)> = Vec::new();
                    for (p_idx, &i) in valid_ps.iter().enumerate() {
                        let k = i - 1;
                        let score = m1_scores[p_idx] + self.sorted_best_m[k as usize][0].0;
                        heap.push((score, p_idx, 0));
                    }
                    let mut heap = std::collections::BinaryHeap::from(heap);

                    let mut filled = 0usize;
                    let mut prev_score = SCORE_MIN;
                    let mut current_score = SCORE_MIN;
                    while (filled < self.beam || current_score == prev_score) && !heap.is_empty() {
                        let (score, p_idx, m_idx) = heap.pop().unwrap();
                        prev_score = current_score;
                        current_score = score;
                        let i = valid_ps[p_idx];
                        let k = i - 1;
                        let newi = self.sorted_best_m[k as usize][m_idx].1;
                        let newscore = m1_scores[p_idx] + self.best_m[k as usize][&newi].score;

                        let entry = self.best_m2[j].entry(newi).or_default();
                        if matches!(entry.manner, Manner::None) {
                            filled += 1;
                            entry.update_if_better_split(newscore, Manner::M2EqMPlusP, k);
                        }

                        let mut next_m_idx = m_idx + 1;
                        while next_m_idx < self.sorted_best_m[k as usize].len() {
                            let candidate_score =
                                m1_scores[p_idx] + self.sorted_best_m[k as usize][next_m_idx].0;
                            let candidate_newi = self.sorted_best_m[k as usize][next_m_idx].1;
                            if !self.best_m2[j].contains_key(&candidate_newi) {
                                heap.push((candidate_score, p_idx, next_m_idx));
                                break;
                            } else {
                                next_m_idx += 1;
                            }
                        }
                    }
                }

                self.best_c[j] = beamstep_c;
                self.best_p[j] = beamstep_p;
            }

            // Beam of M2
            {
                let mut beamstep_m2 = std::mem::take(&mut self.best_m2[j]);

                if self.beam > 0 && beamstep_m2.len() > self.beam {
                    beam_prune(&mut beamstep_m2, &self.best_c, &mut self.scores, self.beam);
                }

                sort_keys(&beamstep_m2, &mut self.keys);
                let keys: Vec<(i32, State)> = self.keys.clone();
                for (i, state) in keys.iter() {
                    let i = *i as usize;

                    // M = M2
                    {
                        self.best_m[j]
                            .entry(i as i32)
                            .or_default()
                            .update_if_better(state.score, Manner::MEqM2);
                    }

                    // multi-loop
                    {
                        let start_p = i.saturating_sub(SINGLE_MAX_LEN);
                        for p in (start_p..i).rev() {
                            let nucp = self.nucs[p];
                            let mut q = next_pair[nucp][j];

                            while q != -1
                                && (i - p - 1 <= SINGLE_MAX_LEN)
                                && self.pair_span_allowed(p, q as usize)
                            {
                                let qu = q as usize;
                                let newscore =
                                    state.score - score_multi_unpaired() - score_multi_unpaired();
                                self.best_multi[qu]
                                    .entry(p as i32)
                                    .or_default()
                                    .update_if_better_paddings(
                                        newscore,
                                        Manner::Multi,
                                        (i - p) as i8,
                                        (qu - j) as i32,
                                    );
                                q = next_pair[nucp][qu];
                            }
                        }
                    }
                }

                self.best_m2[j] = beamstep_m2;
            }

            // Beam of M
            {
                let mut beamstep_m = std::mem::take(&mut self.best_m[j]);

                let threshold = if self.beam > 0 && beamstep_m.len() > self.beam {
                    beam_prune(&mut beamstep_m, &self.best_c, &mut self.scores, self.beam)
                } else {
                    SCORE_MIN
                };

                sort_m(
                    threshold,
                    &beamstep_m,
                    &mut self.sorted_best_m[j],
                    &self.best_c,
                    &self.scores,
                );

                sort_keys(&beamstep_m, &mut self.keys);
                let keys: Vec<(i32, State)> = self.keys.clone();
                for (i, state) in keys.iter() {
                    let i = *i as usize;
                    if j < seq_length - 1 {
                        let newscore = state.score - score_multi_unpaired();
                        self.best_m[j + 1]
                            .entry(i as i32)
                            .or_default()
                            .update_if_better(newscore, Manner::MEqMPlusU);
                    }
                }

                self.best_m[j] = beamstep_m;
            }

            // Beam of C
            {
                if j < seq_length - 1 {
                    let newscore = self.best_c[j].score - score_external_unpaired();
                    self.best_c[j + 1].update_if_better(newscore, Manner::CEqCPlusU);
                }
            }
        }

        let viterbi = self.best_c[seq_length - 1];
        let structure = self.get_parentheses(seq);

        DecoderResult {
            structure,
            score: viterbi.score,
        }
    }
}

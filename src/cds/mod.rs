use crate::beam_search::SearchPath;
use crate::cai::{cai_loss_term, tri2aa, CaiMode, CodonTable};
use crate::codon_pll_binding::call_codon_pll::{cds_to_codon_ids, eval_pll};
use crate::gc::gc_optimizer::GcOptimizer;
use crate::mfe_binding::call_mfe::{compute_mfe, MfeMethod};
use crate::mutate2stop::mutate2stop_numbers;
use crate::palindrome::palindrome_score;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use std::fmt;

/// A struct for triplet chars
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Triplet {
    pub triplet: [char; 3],
}

impl Triplet {
    pub fn new(t1: char, t2: char, t3: char) -> Self {
        Triplet {
            triplet: [t1, t2, t3],
        }
    }

    /// The calller should make sure the size of input seq is 3 char
    pub fn from_string(seq: String) -> Self {
        let seq_c = seq.chars().collect::<Vec<_>>();
        Triplet {
            triplet: [seq_c[0], seq_c[1], seq_c[2]],
        }
    }

    pub fn to_string(&self) -> String {
        String::from_iter(self.triplet)
    }
}

#[derive(Clone)]
pub struct CDSSeq {
    /// A CDS sequence from input
    pub raw_seq: Vec<char>,
    /// The triplet sequence of the CDS
    pub triplet_seq: VecDeque<Triplet>,
    /// The amino acids sequence of the CDS
    pub aa_seq: VecDeque<String>,
}

impl CDSSeq {
    pub fn new<T: AsRef<str>>(in_seq: T) -> Self {
        let mut cds = CDSSeq {
            raw_seq: in_seq.as_ref().chars().into_iter().collect(),
            triplet_seq: VecDeque::new(),
            aa_seq: VecDeque::new(),
        };
        cds.tripleting();
        cds
    }

    /// Get the number of triplets/amino_acids
    pub fn triplet_num(&self) -> usize {
        self.triplet_seq.len()
    }

    pub fn get_cds(&self) -> String {
        let mut outs = Vec::new();
        for t in &self.triplet_seq {
            outs.push(t.to_string());
        }
        outs.join("")
    }

    /// To check if the length of a given sequence is a multiple of 3
    pub fn check_size(s: &str) -> bool {
        s.len() % 3 == 0
    }

    /// To check if all the bases in a CDS seq are in [`A` `T` `C` `G`]
    pub fn check_base(s: &str) -> bool {
        let mut dna_bases: HashSet<char> = HashSet::new();
        for i in vec!['A', 'T', 'C', 'G'] {
            dna_bases.insert(i);
        }
        let bases = s.chars().into_iter().collect::<HashSet<char>>();
        for i in bases {
            if !dna_bases.contains(&i) {
                return false;
            }
        }
        return true;
    }

    /// To check if all the triplets in a CDS are standard triplets
    pub fn check_triplets(s: &str) -> bool {
        let bytes = s.as_bytes();
        for i in (0..bytes.len()).step_by(3) {
            if tri2aa(std::str::from_utf8(&bytes[i..i + 3]).unwrap()).is_empty() {
                return false;
            }
        }
        true
    }
}

impl CDSSeq {
    /// Generate the triplet sequence and the amino acids sequence
    fn tripleting(&mut self) {
        let mut aa_seq = VecDeque::new();
        for win in self.raw_seq.chunks(3) {
            let triplet = Triplet::new(win[0], win[1], win[2]);
            aa_seq.push_back(tri2aa(&triplet.to_string()[..]));
            self.triplet_seq.push_back(triplet);
        }
        self.aa_seq = aa_seq;
    }
}

use std::cmp::Ordering;

/// A struct to store the loss values
#[derive(Clone, Serialize)]
pub struct LossInfo {
    /// The final loss value
    pub loss: f64,
    /// The mfe part
    pub mfe: f64,
    /// The scaled cai part
    pub scaled_cai: f64,
    /// The secondary structure of RNA sequence
    pub second: String,
    /// The palindrome score
    pub palindrome_score: f64,
    pub palindrome_seqs: Vec<String>,
    /// The mutate2stop score
    pub m2s_score: f64,
    /// The number of codons which could easily mutate to stop codons
    pub m2s_nums: u64,
    /// The codon PLL score (negated * weight, for unified optimization direction)
    pub pll_score: f64,
    /// The raw codon PLL value as returned by the CodonTransformer model
    /// (no negation, no weight — the original pseudo-log-likelihood).
    /// This is the user-facing value, analogous to raw GC%.
    pub raw_pll: f64,
}

impl LossInfo {
    fn new(loss: f64, mfe: f64, scaled_cai: f64, second: String) -> Self {
        LossInfo {
            loss,
            mfe,
            scaled_cai,
            second,
            palindrome_score: 0.0,
            palindrome_seqs: Vec::new(),
            m2s_score: 0.0,
            m2s_nums: 0,
            pll_score: 0.0,
            raw_pll: 0.0,
        }
    }

    fn with_palindrome(mut self, score: f64, seqs: Vec<String>) -> Self {
        self.palindrome_score = score;
        self.palindrome_seqs = seqs;
        self
    }

    fn with_m2s(mut self, score: f64, nums: u64) -> Self {
        self.m2s_score = score;
        self.m2s_nums = nums;
        self
    }

    fn with_pll(mut self, pll_score: f64, raw_pll: f64) -> Self {
        self.pll_score = pll_score;
        self.raw_pll = raw_pll;
        self
    }
}

impl fmt::Display for LossInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LossInfo(loss: {}, mfe: {}, scaled_cai: {})",
            self.loss, self.mfe, self.scaled_cai
        )
    }
}

impl PartialOrd for LossInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.loss.partial_cmp(&other.loss)
    }
}

impl PartialEq for LossInfo {
    fn eq(&self, other: &Self) -> bool {
        self.loss == other.loss
    }
}

/// The loss functions
pub struct LossFuncs;

impl LossFuncs {
    /// The cai-mfe loss function.
    /// The argument `lambda` is a hyperparameter to
    /// balance the weights of MFE and CAI when lambda is zero
    /// it means the loss function only consider the MFE part
    /// check the `LinearDesign` paper for detail
    pub fn loss_cai_mfe(
        sp: &SearchPath,
        lambda: f64,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode);
        // to calculate mfe the T in CDS needs to be replaced with U
        let rna = sp.get_rna();
        let (mfe_part, second) = if skip_mfe_pll {
            (0.0, "".to_string())
        } else {
            compute_mfe(rna, mfe_method)
        };
        let loss = mfe_part + (lambda * scaled_cai);
        LossInfo::new(loss, mfe_part, scaled_cai, second)
    }

    /// Note, for the LossInfo here the only valid value is `loss`
    pub fn loss_cai_gc(sp: &SearchPath, table: &'static CodonTable, cai_mode: CaiMode) -> LossInfo {
        let seq = sp.get_cds();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode);
        let scaled_gc = sp.get_gc_content() * -1.0 * seq.len() as f64;
        LossInfo::new(scaled_gc + scaled_cai, 0.0, 0.0, "".into())
    }

    /// The loss function that calculate mfe, cai and gc at the same time
    pub fn loss_cai_mfe_gc(
        lambda: f64,
        sp: &SearchPath,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let scaled_cai = cai_loss_term(&sp.get_cds(), table, cai_mode);
        // to calculate mfe the T in CDS needs to be replaced with U
        let rna = sp.get_rna();
        let (mfe_part, second) = if skip_mfe_pll {
            (0.0, "".to_string())
        } else {
            compute_mfe(rna, mfe_method)
        };
        let gc_optimizer = GcOptimizer::new(sp);
        let scaled_gc = gc_optimizer.gc_abs_loss() * sp.get_cds_len() as f64; // negative
        let loss = mfe_part + lambda * (scaled_cai + scaled_gc);
        LossInfo::new(loss, mfe_part, scaled_cai, second)
    }

    pub fn loss_cai_mfe_palin(
        sp: &SearchPath,
        lambda: f64,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        weight_cai: f64,
        weight_palin: f64,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode) * seq.len() as f64 * weight_cai;
        let rna = sp.get_rna();
        let (palin_score, palin_seqs) = palindrome_score(&rna);
        let scaled_palin_score = palin_score * seq.len() as f64 * weight_palin; // negative
        let (mfe_part, second) = if skip_mfe_pll {
            (0.0, "".to_string())
        } else {
            compute_mfe(rna, mfe_method)
        };
        let loss = mfe_part + lambda * (scaled_cai + scaled_palin_score);
        LossInfo::new(loss, mfe_part, scaled_cai, second)
            .with_palindrome(scaled_palin_score, palin_seqs)
    }

    pub fn loss_cai_gc_palin(
        sp: &SearchPath,
        table: &'static CodonTable,
        weight_cai: f64,
        weight_gc: f64,
        weight_palin: f64,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let rna = sp.get_rna();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode) * seq.len() as f64 * weight_cai;
        let scaled_gc = sp.get_gc_content() * -1.0 * seq.len() as f64 * weight_gc; // negative
        let (palin_score, palin_seqs) = palindrome_score(&rna);
        let scaled_palin_score = palin_score * seq.len() as f64 * weight_palin; // negative
        let loss = scaled_gc + scaled_cai + scaled_palin_score; // negative
        LossInfo::new(loss, 0.0, 0.0, "".into()).with_palindrome(scaled_palin_score, palin_seqs)
    }

    pub fn loss_cai_mfe_gc_palin(
        lambda: f64,
        sp: &SearchPath,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        weight_cai: f64,
        weight_gc: f64,
        weight_palin: f64,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode) * seq.len() as f64 * weight_cai;
        // to calculate mfe the T in CDS needs to be replaced with U
        let rna = sp.get_rna();
        let (palin_score, palin_seqs) = palindrome_score(&rna);
        let scaled_palin_score = palin_score * seq.len() as f64 * weight_palin;
        let (mfe_part, second) = if skip_mfe_pll {
            (0.0, "".to_string())
        } else {
            compute_mfe(rna, mfe_method)
        };
        let gc_optimizer = GcOptimizer::new(sp);
        let scaled_gc = gc_optimizer.gc_abs_loss() * seq.len() as f64 * weight_gc;
        let loss = mfe_part + lambda * (scaled_cai + scaled_gc + scaled_palin_score);
        LossInfo::new(loss, mfe_part, scaled_cai, second)
            .with_palindrome(scaled_palin_score, palin_seqs)
    }

    // For deoptimization

    pub fn loss_cai_mfe_gc_m2s_palin(
        lambda: f64,
        sp: &SearchPath,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        weight_cai: f64,
        weight_gc: f64,
        weight_m2s: f64,
        weight_palin: f64,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let scaled_cai = cai_loss_term(&seq, table, cai_mode) * seq.len() as f64 * weight_cai;
        let rna = sp.get_rna();
        let (palin_score, palin_seqs) = palindrome_score(&rna);
        let scaled_palin_score = palin_score * seq.len() as f64 * weight_palin;
        let (mfe_part, second) = if skip_mfe_pll {
            (0.0, "".to_string())
        } else {
            compute_mfe(rna, mfe_method)
        };
        let gc_optimizer = GcOptimizer::new(sp);
        let scaled_gc = gc_optimizer.gc_abs_loss() * seq.len() as f64 * weight_gc;
        let m2s_num = mutate2stop_numbers(&seq);
        let scaled_m2s = m2s_num * weight_m2s;
        let loss = mfe_part + lambda * (scaled_cai + scaled_gc + scaled_m2s + scaled_palin_score);
        LossInfo::new(loss, mfe_part, scaled_cai, second)
            .with_palindrome(scaled_palin_score, palin_seqs)
            .with_m2s(scaled_m2s, m2s_num as u64)
    }

    pub fn loss_cai_gc_m2s_palin(
        lambda: f64,
        sp: &SearchPath,
        table: &'static CodonTable,
        weight_cai: f64,
        weight_m2s: f64,
        weight_gc: f64,
        weight_palin: f64,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let rna = sp.get_rna();
        let (parline_score, palin_seqs) = palindrome_score(&rna);
        let scaled_parline_score = parline_score * seq.len() as f64 * weight_palin;
        let scaled_cai = cai_loss_term(&seq, table, cai_mode) * seq.len() as f64 * weight_cai;
        let gc_optimizer = GcOptimizer::new(sp);
        let scaled_gc = gc_optimizer.gc_abs_loss() * seq.len() as f64 * weight_gc;
        let m2s_num = mutate2stop_numbers(&seq);
        let scaled_m2s = m2s_num * weight_m2s;
        let loss = lambda * (scaled_cai + scaled_m2s + scaled_parline_score + scaled_gc);
        LossInfo::new(loss, 0.0, scaled_cai, "".into())
            .with_palindrome(parline_score, palin_seqs)
            .with_m2s(scaled_m2s, m2s_num as u64)
    }

    // Unified loss function

    /// The unified loss function. If the weight_mfe is zero the MFE will not be calculated
    pub fn loss_unified(
        lambda: f64,
        sp: &SearchPath,
        table: &'static CodonTable,
        mfe_method: MfeMethod,
        weight_mfe: i64,
        weight_cai: f64,
        weight_gc: f64,
        weight_m2s: f64,
        weight_palin: f64,
        model_path: Option<&str>,
        weight_pll: f64,
        skip_mfe_pll: bool,
        cai_mode: CaiMode,
    ) -> LossInfo {
        let seq = sp.get_cds();
        let raw_cai = cai_loss_term(&seq, table, cai_mode);
        let rna = sp.get_rna();
        let (palin_score, palin_seqs) = palindrome_score(&rna);
        // native direction: fewer is better → keep positive, no negation
        let raw_palin = palin_score * seq.len() as f64;
        let gc_optimizer = GcOptimizer::new(sp);
        // raw_gc is negative already
        let raw_gc = gc_optimizer.gc_abs_loss() * seq.len() as f64;
        let raw_m2s_num = mutate2stop_numbers(&seq);
        // native direction: fewer is better → keep positive, no negation
        let raw_m2s_score = raw_m2s_num;

        // Calculate codon PLL if model path is provided and weight is non-zero
        // and we are past the skip threshold
        let (raw_pll_loss, raw_pll_display) =
            if weight_pll != 0.0 && model_path.is_some() && !skip_mfe_pll {
                let seq_ids = cds_to_codon_ids(&seq);
                let pll = eval_pll(model_path.unwrap(), &seq_ids);
                // PLL is a log-likelihood (usually negative); higher is better.
                // To make it consistent with other terms (lower loss is better), negate it
                // for the loss calculation (`raw_pll_loss`).
                // Keep the original model output as `raw_pll_display` for user-facing reporting.
                (-1.0 * pll, pll)
            } else {
                (0.0, 0.0)
            };

        let mut mfe_part: f64 = 0.0;
        let mut second: String = "".to_string();
        let should_calc_mfe = weight_mfe != 0 && !skip_mfe_pll;
        if should_calc_mfe {
            let scaled_cai = raw_cai * weight_cai;
            let scaled_palin_score = raw_palin * weight_palin;
            let scaled_gc = raw_gc * weight_gc;
            let scaled_m2s = raw_m2s_score * weight_m2s;
            let scaled_pll = raw_pll_loss * weight_pll;

            let (raw_mfe, raw_second) = compute_mfe(rna, mfe_method);
            // weight_mfe only affects the loss function, not the reported MFE value
            mfe_part = raw_mfe * weight_mfe as f64;
            second = raw_second;

            let loss = mfe_part
                + lambda * (scaled_cai + scaled_gc + scaled_m2s + scaled_palin_score + scaled_pll);
            return LossInfo::new(loss, raw_mfe, scaled_cai, second)
                .with_palindrome(scaled_palin_score, palin_seqs)
                .with_m2s(scaled_m2s, raw_m2s_num as u64)
                .with_pll(scaled_pll, raw_pll_display);
        } else {
            // NOTE: without MFE the raw terms are used directly instead of the
            // lambda-scaled variants.  However PLL must still be scaled by
            // lambda so that its contribution to the loss is consistent with
            // the MFE branch.  The reported pll_score is kept as
            // raw_pll_loss * weight_pll (without lambda) for consistency with the
            // MFE-branch convention.
            let scaled_pll = raw_pll_loss * weight_pll;
            let loss = raw_cai + raw_gc + raw_m2s_score + raw_palin + lambda * scaled_pll;
            return LossInfo::new(loss, mfe_part, raw_cai, second)
                .with_palindrome(raw_palin, palin_seqs)
                .with_m2s(raw_m2s_score, raw_m2s_num as u64)
                .with_pll(scaled_pll, raw_pll_display);
        }
    }
}

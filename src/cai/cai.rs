use crate::beam_search::CodonTables;
use crate::cai::table::{codon_to_idx, CodonTable, AA_CHARS, AA_OF_CODON};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;

/// Calculate the geometric-mean CAI for a given CDS sequence.
///
/// The "raw" CAI is the geometric mean of `freq(codon) / max_freq(aa)` over
/// all codons, so the result lies in `[0, 1]`.
pub(crate) fn call_raw_cai(seq: &str, table: &CodonTable) -> f64 {
    let bytes = seq.as_bytes();
    let n = bytes.len() / 3;
    let mut cai_out: f64 = 1.0;
    for i in (0..bytes.len()).step_by(3) {
        let ratio = table.cai_ratio(&bytes[i..i + 3]);
        cai_out *= ratio;
    }
    cai_out.powf(1.0 / n as f64)
}

/// Calculate the arithmetic-mean CAI for a given CDS sequence.
///
/// Using the arithmetic mean avoids the geometric mean collapsing to zero when
/// rare codons are present.
pub(crate) fn call_raw_arithmetic_cai(seq: &str, table: &CodonTable) -> f64 {
    let bytes = seq.as_bytes();
    let n = bytes.len() / 3;
    let mut cai_out: f64 = 0.0;
    for i in (0..bytes.len()).step_by(3) {
        let ratio = table.cai_ratio(&bytes[i..i + 3]);
        cai_out += ratio;
    }
    cai_out / n as f64
}

/// Calculate the LinearDesign-style scaled CAI: `-Σ ln(freq / max_freq)`.
pub(crate) fn call_scaled_cai(seq: &str, table: &CodonTable) -> f64 {
    let bytes = seq.as_bytes();
    let mut cai_out: f64 = 0.0;
    for i in (0..bytes.len()).step_by(3) {
        let ratio = table.cai_ratio(&bytes[i..i + 3]);
        cai_out -= ratio.ln();
    }
    cai_out
}

/// Available CAI variants used during optimization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaiMode {
    /// LinearDesign-style scaled CAI: `-Σ ln(freq/max_freq)`.
    /// This is the default for the `fit` command.
    #[serde(rename = "scaled")]
    #[default]
    Scaled,
    /// Arithmetic mean of codon adaptation ratios.
    /// This is the default for the `ufit` command.
    #[serde(rename = "arithmetic")]
    Arithmetic,
    /// Traditional geometric-mean CAI.
    #[serde(rename = "geometric")]
    Geometric,
}

impl FromStr for CaiMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "scaled" => Ok(CaiMode::Scaled),
            "arithmetic" => Ok(CaiMode::Arithmetic),
            "geometric" => Ok(CaiMode::Geometric),
            _ => Err(format!(
                "unknown CAI mode '{}'; expected 'scaled', 'arithmetic', or 'geometric'",
                s
            )),
        }
    }
}

/// Return the CAI term used in the optimization loss for the selected mode.
///
/// Lower values are better. For `Arithmetic` and `Geometric` the value is
/// negated and scaled by sequence length so that it is comparable to MFE/GC
/// terms in the loss function.
pub fn cai_loss_term(seq: &str, table: &CodonTable, mode: CaiMode) -> f64 {
    match mode {
        CaiMode::Scaled => call_scaled_cai(seq, table),
        CaiMode::Arithmetic => -1.0 * call_raw_arithmetic_cai(seq, table) * seq.len() as f64,
        CaiMode::Geometric => -1.0 * call_raw_cai(seq, table) * seq.len() as f64,
    }
}

/// Convert a DNA triplet to its one-letter amino-acid code.
///
/// The caller must ensure the input is a valid DNA codon.
pub(crate) fn tri2aa(triplet: &str) -> String {
    let idx = codon_to_idx(triplet.as_bytes());
    AA_CHARS[AA_OF_CODON[idx as usize] as usize].to_string()
}

/// Calculator that evaluates all three CAI variants for a single sequence.
pub struct RawCaiCalculator<'a> {
    seq_id: &'a String,
    seq: &'a String,
    cai_value: Option<f64>,
    arithmetic_cai: Option<f64>,
    scaled_cai: Option<f64>,
    codon_table: CodonTables,
}

impl<'a> RawCaiCalculator<'a> {
    pub fn new(seq_id: &'a String, seq: &'a String, table: CodonTables) -> Self {
        let mut out = RawCaiCalculator {
            seq_id,
            seq,
            cai_value: None,
            arithmetic_cai: None,
            scaled_cai: None,
            codon_table: table,
        };
        out.cai();
        out
    }

    /// Calculate the three CAI variants for the stored sequence.
    fn cai(&mut self) {
        let table = self.codon_table.get_codon_table();
        self.cai_value = Some(call_raw_cai(&self.seq[..], table));
        self.arithmetic_cai = Some(call_raw_arithmetic_cai(&self.seq[..], table));
        self.scaled_cai = Some(call_scaled_cai(&self.seq[..], table));
    }

    /// Convert the result to a JSON value after calling `cai`.
    pub fn to_json(&self) -> Value {
        json!({
            "seq_id": self.seq_id.clone(),
            "cai": self.cai_value.unwrap(),
            "scaled_cai": self.scaled_cai.unwrap(),
            "arithmetic_cai": self.arithmetic_cai.unwrap()
        })
    }
}

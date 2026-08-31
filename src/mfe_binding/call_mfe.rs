use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Available MFE calculation backends.
///
/// The C++ LinearFold backend has been removed; only the pure-Rust port remains.
/// The `cpp` value is kept as a deprecated alias for backward compatibility with
/// existing TOML configuration files and scripts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MfeMethod {
    /// Deprecated alias that now maps to the Rust LinearFold implementation.
    #[serde(rename = "cpp")]
    Linear,
    /// Pure-Rust LinearFold port.
    #[serde(rename = "rust")]
    LinearRust,
}

impl Default for MfeMethod {
    fn default() -> Self {
        MfeMethod::LinearRust
    }
}

impl FromStr for MfeMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpp" | "c++" | "linear" => Ok(MfeMethod::Linear),
            "rust" | "linear-rust" | "linearrust" => Ok(MfeMethod::LinearRust),
            _ => Err(format!(
                "unknown MFE method '{}'; expected 'rust' (or deprecated 'cpp')",
                s
            )),
        }
    }
}

/// Sentinel value stored in the atomic to mean "no maximum pair distance".
const NO_MAX_PAIR_DIST: usize = usize::MAX;

static MFE_MAX_PAIR_DIST: AtomicUsize = AtomicUsize::new(NO_MAX_PAIR_DIST);

/// Set the maximum base-pair span used by subsequent MFE calculations.
///
/// `None` disables the limit (global fold). This is intentionally a global
/// runtime knob so that the large call graph between `cosyn` and the loss
/// functions does not need to be threaded with an extra parameter during the
/// initial prototyping phase.
pub fn set_mfe_max_pair_dist(dist: Option<usize>) {
    MFE_MAX_PAIR_DIST.store(dist.unwrap_or(NO_MAX_PAIR_DIST), Ordering::Relaxed);
}

fn get_mfe_max_pair_dist() -> Option<usize> {
    let v = MFE_MAX_PAIR_DIST.load(Ordering::Relaxed);
    if v == NO_MAX_PAIR_DIST {
        None
    } else {
        Some(v)
    }
}

/// Dispatch an MFE calculation to the Rust LinearFold backend.
///
/// The C++ backend has been removed, so both enum variants now resolve to the
/// same Rust implementation.
///
/// The optional maximum base-pair span is read from a global runtime setting
/// (`set_mfe_max_pair_dist`). `None` performs a full global fold (default).
pub fn compute_mfe<T: AsRef<str>>(seq: T, _method: MfeMethod) -> (f64, String) {
    let max_pair_dist = get_mfe_max_pair_dist();
    linearfold_rust::rna_linear_mfe_with_options(seq.as_ref(), 100, true, max_pair_dist)
}

pub struct MfeCalculator<'a> {
    seq_id: &'a String,
    seq: &'a String,
    mfe_results: Option<(f64, String)>,
}

impl<'a> MfeCalculator<'a> {
    pub fn new(seq_id: &'a String, seq: &'a String, mfe_method: MfeMethod) -> Self {
        let mut out = MfeCalculator {
            seq_id,
            seq,
            mfe_results: None,
        };
        out.linear_mfe(mfe_method);
        out
    }

    /// Calculate MFE with the Rust LinearFold backend.
    fn linear_mfe(&mut self, mfe_method: MfeMethod) {
        let rna = self.seq.replace("T", "U");
        self.mfe_results = Some(compute_mfe(rna, mfe_method));
    }

    pub fn to_json(&self) -> Value {
        let (mfe_value, second) = self.mfe_results.clone().unwrap();
        json!({
            "seq_id": self.seq_id.clone(),
            "raw_mfe": mfe_value,
            "secondary": second,
        })
    }
}

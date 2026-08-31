#[cfg(feature = "pll")]
use std::ffi::CString;
#[cfg(feature = "pll")]
use std::os::raw::{c_char, c_double, c_int, c_void};
#[cfg(not(feature = "pll"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "pll")]
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

#[cfg(feature = "pll")]
unsafe extern "C" {
    fn codon_batch_service_init(
        path: *const c_char,
        max_batch_size: c_int,
        timeout_ms: c_int,
    ) -> *mut c_void;
    fn codon_batch_service_free(service: *mut c_void);
    fn codon_batch_evaluate(service: *mut c_void, seq: *const i64, len: c_int) -> c_double;
}

#[cfg(feature = "pll")]
/// Global singleton batch inference service pointer.
/// Initialized lazily on first `eval_pll` call via CAS.
static BATCH_SERVICE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

#[cfg(feature = "pll")]
/// Batch size and timeout used when creating the global BatchService.
/// These are set via `configure_pll_batch_params()` *before* the first PLL call.
/// Changing them after the service is initialized has no effect.
static PLL_BATCH_SIZE: AtomicUsize = AtomicUsize::new(256);

#[cfg(feature = "pll")]
static PLL_TIMEOUT_MS: AtomicUsize = AtomicUsize::new(50);

#[cfg(not(feature = "pll"))]
/// Used to emit a single runtime warning when PLL is requested but disabled.
static PLL_DISABLED_WARNED: AtomicBool = AtomicBool::new(false);

/// Configure the GPU batch accumulation parameters.
///
/// Must be called **before** the first `eval_pll` call (i.e. early in
/// `run_ufit` / `run_fit`).  Once the C++ BatchService is created these
/// values are baked in and further calls are ignored.
///
/// * `batch_size` – max sequences per GPU inference batch (default 256).
/// * `timeout_ms` – how long the background thread waits to accumulate a
///   batch before processing whatever it has (default 50 ms).
#[cfg(feature = "pll")]
pub fn configure_pll_batch_params(batch_size: usize, timeout_ms: usize) {
    PLL_BATCH_SIZE.store(batch_size, Ordering::Relaxed);
    PLL_TIMEOUT_MS.store(timeout_ms, Ordering::Relaxed);
}

#[cfg(not(feature = "pll"))]
pub fn configure_pll_batch_params(_batch_size: usize, _timeout_ms: usize) {
    if !PLL_DISABLED_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[codon_pll] Warning: cosyn was built without PLL support (--no-default-features). PLL scores will be reported as 0.0."
        );
    }
}

#[cfg(feature = "pll")]
/// Initialize the global batch service if not already done.
fn ensure_service_init(model_path: &str) -> *mut c_void {
    let ptr = BATCH_SERVICE.load(Ordering::Acquire);
    if !ptr.is_null() {
        return ptr;
    }

    let batch_size = PLL_BATCH_SIZE.load(Ordering::Relaxed) as c_int;
    let timeout_ms = PLL_TIMEOUT_MS.load(Ordering::Relaxed) as c_int;
    let c_path = CString::new(model_path).expect("Failed to convert model path to C string");
    let new_ptr = unsafe { codon_batch_service_init(c_path.as_ptr(), batch_size, timeout_ms) };
    if new_ptr.is_null() {
        panic!(
            "[codon_pll] Failed to initialize batch service: {}",
            model_path
        );
    }

    match BATCH_SERVICE.compare_exchange(
        std::ptr::null_mut(),
        new_ptr,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => new_ptr,
        Err(actual) => {
            // Another thread won the race; free our instance and use theirs.
            unsafe {
                codon_batch_service_free(new_ptr);
            }
            actual
        }
    }
}

/// Maximum number of tokens the CodonTransformer PLL model can accept.
/// The C++ BatchService rejects sequences longer than this value.
/// A CDS of N codons is encoded as [CLS] + N codon tokens + __unk + [SEP],
/// i.e. N + 3 tokens. Therefore the effective limits are:
///   - max codons (including stop codon, if present): 2045
///   - max amino-acid length (assuming a trailing stop codon): ~2044
pub const MAX_PLL_TOKEN_LEN: usize = 2048;

/// Maximum number of codons (including any stop codon) that can be fed to PLL.
pub const MAX_PLL_CODON_COUNT: usize = MAX_PLL_TOKEN_LEN - 3; // 2045

/// Conservative maximum protein length in amino acids for PLL input.
/// This assumes the CDS includes a stop codon. Without a stop codon the
/// limit is MAX_PLL_CODON_COUNT.
pub const MAX_PLL_AA_LEN: usize = MAX_PLL_CODON_COUNT - 1; // 2044

/// Evaluate the pseudo-log-likelihood (PLL) of a codon sequence.
/// Requests from multiple Rust threads are automatically batched on the GPU
/// by the C++ BatchService background thread.
#[cfg(feature = "pll")]
pub fn eval_pll(model_path: &str, seq: &[i64]) -> f64 {
    if seq.len() > MAX_PLL_TOKEN_LEN {
        eprintln!(
            "[codon_pll] Warning: input length {} tokens exceeds the PLL limit of {} \
             (~{} amino acids with stop codon). PLL score set to 0.0.",
            seq.len(),
            MAX_PLL_TOKEN_LEN,
            MAX_PLL_AA_LEN
        );
        return 0.0;
    }
    let service = ensure_service_init(model_path);
    let result = unsafe { codon_batch_evaluate(service, seq.as_ptr(), seq.len() as c_int) };
    if result.is_nan() {
        eprintln!(
            "[codon_pll] Warning: model returned NaN for sequence length {}, \
             returning a large penalty so the optimizer avoids this sequence.",
            seq.len()
        );
        // Return a very negative PLL value so the optimizer avoids sequences
        // that cause NaN (e.g. numerical overflow).  This is safer than
        // silently returning 0.0 which would make the sequence look optimal.
        -1e9
    } else {
        result
    }
}

#[cfg(not(feature = "pll"))]
pub fn eval_pll(_model_path: &str, _seq: &[i64]) -> f64 {
    0.0
}

/// Shut down the global batch service and release GPU resources.
/// Safe to call multiple times; only the first call has effect.
#[cfg(feature = "pll")]
#[allow(dead_code)]
pub fn shutdown_batch_service() {
    let ptr = BATCH_SERVICE.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if !ptr.is_null() {
        unsafe {
            codon_batch_service_free(ptr);
        }
    }
}

#[cfg(not(feature = "pll"))]
#[allow(dead_code)]
pub fn shutdown_batch_service() {}

/// Convert a DNA CDS string (length must be a multiple of 3) into a vector of token IDs
/// compatible with the CodonTransformer model.
///
/// The returned sequence has the format:
/// [CLS] <aa_codon_1> <aa_codon_2> ... <aa_codon_N> [SEP]
///
/// where each token is in the form `{amino_acid}_{codon}`.
/// Stop codons use `__` as the amino acid prefix (e.g. `__TAA`).
pub fn cds_to_codon_ids(cds: &str) -> Vec<i64> {
    let chars: Vec<char> = cds.chars().collect();
    assert_eq!(chars.len() % 3, 0, "CDS length must be a multiple of 3");

    let mut ids = Vec::with_capacity(chars.len() / 3 + 3);
    ids.push(1); // [CLS]

    for chunk in chars.chunks(3) {
        let codon: String = chunk.iter().collect();
        let aa = crate::cai::tri2aa(&codon);
        let id = aa_codon_to_id(&aa, &codon);
        ids.push(id);
    }

    ids.push(25); // __unk (extra stop/unknown token added by tokenizer)
    ids.push(2); // [SEP]
    ids
}

/// Map an amino acid (single-letter or "*") and its codon to the CodonTransformer vocab ID.
fn aa_codon_to_id(aa: &str, codon: &str) -> i64 {
    match (aa, codon) {
        // A
        ("A", "GCT") => 65,
        ("A", "GCC") => 63,
        ("A", "GCA") => 62,
        ("A", "GCG") => 64,
        // C
        ("C", "TGT") => 85,
        ("C", "TGC") => 83,
        // D
        ("D", "GAT") => 61,
        ("D", "GAC") => 59,
        // E
        ("E", "GAA") => 58,
        ("E", "GAG") => 60,
        // F
        ("F", "TTT") => 89,
        ("F", "TTC") => 87,
        // G
        ("G", "GGT") => 69,
        ("G", "GGC") => 67,
        ("G", "GGA") => 66,
        ("G", "GGG") => 68,
        // H
        ("H", "CAT") => 45,
        ("H", "CAC") => 43,
        // I
        ("I", "ATT") => 41,
        ("I", "ATC") => 39,
        ("I", "ATA") => 38,
        // K
        ("K", "AAA") => 26,
        ("K", "AAG") => 28,
        // L
        ("L", "TTA") => 86,
        ("L", "TTG") => 88,
        ("L", "CTT") => 57,
        ("L", "CTC") => 55,
        ("L", "CTA") => 54,
        ("L", "CTG") => 56,
        // M
        ("M", "ATG") => 40,
        // N
        ("N", "AAT") => 29,
        ("N", "AAC") => 27,
        // P
        ("P", "CCT") => 49,
        ("P", "CCC") => 47,
        ("P", "CCA") => 46,
        ("P", "CCG") => 48,
        // Q
        ("Q", "CAA") => 42,
        ("Q", "CAG") => 44,
        // R
        ("R", "CGT") => 53,
        ("R", "CGC") => 51,
        ("R", "CGA") => 50,
        ("R", "CGG") => 52,
        ("R", "AGA") => 34,
        ("R", "AGG") => 36,
        // S
        ("S", "TCT") => 81,
        ("S", "TCC") => 79,
        ("S", "TCA") => 78,
        ("S", "TCG") => 80,
        ("S", "AGT") => 37,
        ("S", "AGC") => 35,
        // T
        ("T", "ACT") => 33,
        ("T", "ACC") => 31,
        ("T", "ACA") => 30,
        ("T", "ACG") => 32,
        // V
        ("V", "GTT") => 73,
        ("V", "GTC") => 71,
        ("V", "GTA") => 70,
        ("V", "GTG") => 72,
        // W
        ("W", "TGG") => 84,
        // Y
        ("Y", "TAT") => 77,
        ("Y", "TAC") => 75,
        // Stop
        ("*", "TAA") => 74,
        ("*", "TAG") => 76,
        ("*", "TGA") => 82,
        _ => panic!(
            "Unknown amino acid '{}' or codon '{}' combination",
            aa, codon
        ),
    }
}

/// Calculator for evaluating codon PLL on input CDS sequences.
pub struct CodonPllCalculator<'a> {
    seq_id: &'a String,
    seq: &'a String,
    model_path: String,
    pll_result: Option<f64>,
}

impl<'a> CodonPllCalculator<'a> {
    pub fn new(seq_id: &'a String, seq: &'a String, model_path: String) -> Self {
        let mut out = CodonPllCalculator {
            seq_id,
            seq,
            model_path,
            pll_result: None,
        };
        out.calc_pll();
        out
    }

    fn calc_pll(&mut self) {
        let ids = cds_to_codon_ids(self.seq);
        let pll = eval_pll(&self.model_path, &ids);
        self.pll_result = Some(pll);
    }

    #[allow(dead_code)]
    pub fn get_pll(&self) -> f64 {
        self.pll_result.unwrap()
    }

    pub fn to_json(&self) -> serde_json::Value {
        let pll = self.pll_result.unwrap();
        serde_json::json!({
            "seq_id": self.seq_id.clone(),
            "pll": pll,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cds_to_codon_ids() {
        // MAV_ with DNA ATGGCTGTGTAA
        let ids = cds_to_codon_ids("ATGGCTGTGTAA");
        assert_eq!(ids, vec![1, 40, 65, 72, 74, 25, 2]);
    }
}

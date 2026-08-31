use lazy_static::lazy_static;
use serde_json::{json, Value};
use std::collections::HashSet;

lazy_static! {
    pub(crate) static ref M2S_TB: HashSet<&'static str> = {
        let mut mutate2stop_tb = HashSet::new();
        mutate2stop_tb.insert("TGT");
        mutate2stop_tb.insert("TGC");
        mutate2stop_tb.insert("GAA");
        mutate2stop_tb.insert("GAG");
        mutate2stop_tb.insert("GGA");
        mutate2stop_tb.insert("AAA");
        mutate2stop_tb.insert("AAG");
        mutate2stop_tb.insert("TTA");
        mutate2stop_tb.insert("TTG");
        mutate2stop_tb.insert("CAA");
        mutate2stop_tb.insert("CAG");
        mutate2stop_tb.insert("CGA");
        mutate2stop_tb.insert("AGA");
        mutate2stop_tb.insert("TCA");
        mutate2stop_tb.insert("TCG");
        mutate2stop_tb.insert("TGG");
        mutate2stop_tb.insert("TAT");
        mutate2stop_tb.insert("TAC");
        mutate2stop_tb
    };
}

/// Calculate the score of mutate2stop and the caller must make sure
/// the input is DNA sequence which length is a multiple of 3.
pub(crate) fn mutate2stop_numbers(inputs: &str) -> f64 {
    // The codons that can mutate to stop codons with 1 mutation
    let mutate2stop_tb = &M2S_TB;

    let mut raw_score: f64 = 0.0;
    for codon in inputs.chars().collect::<Vec<char>>().chunks(3) {
        let codon_str = codon.iter().collect::<String>();
        if mutate2stop_tb.contains(&codon_str[..]) {
            raw_score += 1.0;
        }
    }
    //raw_score * (inputs.len() as f64 / 3.0)
    raw_score
}

pub(crate) struct M2sCalculator<'a> {
    seq_id: &'a String,
    dna_seq: &'a String,
}

impl<'a> M2sCalculator<'a> {
    pub(crate) fn new(seq_id: &'a String, dna_seq: &'a String) -> Self {
        M2sCalculator { seq_id, dna_seq }
    }

    pub(crate) fn get_m2s_score(&self) -> f64 {
        mutate2stop_numbers(&self.dna_seq)
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "seq_id": self.seq_id,
            "mutate2stop_nums": self.get_m2s_score(),
        })
    }
}

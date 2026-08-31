use crate::cai::{try_aa_to_idx, CODONS_FOR_AA, CODON_STRINGS};
use rand::Rng;
/// A convertor from amino acids to cds sequence
pub struct AA2CDS<'a> {
    /// Sequence name
    pub seq_id: &'a String,
    /// Input amino acids sequence
    pub aa_seq: &'a String,
}

impl<'a> AA2CDS<'a> {
    pub fn new(seq_id: &'a String, aa_seq: &'a String) -> Self {
        let out = AA2CDS { seq_id, aa_seq };
        if !out.check_amino_acids() {
            println!("ERROR: {} contains non-standard amino acids.", out.seq_id);
            std::process::exit(0);
        }
        out
    }

    /// Convert amino acids to CDS sequence
    pub fn to_cds(&self) -> String {
        let mut cds_seq = Vec::new();
        for c in self.aa_seq.chars() {
            let aa_idx = try_aa_to_idx(&c.to_string()[..]).unwrap();
            let codons = CODONS_FOR_AA[aa_idx as usize];
            let mut rng = rand::thread_rng();
            let target_idx = rng.gen_range(0..codons.len());
            let codon_idx = codons[target_idx];
            cds_seq.push(CODON_STRINGS[codon_idx as usize]);
        }
        cds_seq.join("")
    }

    /// To check if the input amino acids sequences are consists of
    /// standard amino acids, note all the input sequences should be
    /// upper case
    fn check_amino_acids(&self) -> bool {
        for c in self.aa_seq.chars() {
            if try_aa_to_idx(&c.to_string()[..]).is_none() {
                return false;
            }
        }
        true
    }
}

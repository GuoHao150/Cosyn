use std::collections::HashSet;

/// A convertor from RNA to DNA CDS sequence
pub struct RNA2CDS<'a> {
    pub seq_id: &'a String,
    pub rna_seq: &'a String,
}

impl<'a> RNA2CDS<'a> {
    pub fn new(seq_id: &'a String, rna_seq: &'a String) -> Self {
        let out = RNA2CDS { seq_id, rna_seq };
        if !out.check_rna_bases() {
            println!("ERROR: {} sequence is not consist of AUGC", out.seq_id);
            std::process::exit(0);
        }
        out
    }

    /// Convert rna to CDS dna sequence
    pub fn to_cds(&self) -> String {
        self.rna_seq.to_string().replace("U", "T")
    }

    fn check_rna_bases(&self) -> bool {
        let rna_bases = HashSet::from_iter(['A', 'G', 'C', 'U']);
        self.rna_seq
            .to_string()
            .chars()
            .into_iter()
            .collect::<HashSet<char>>()
            .eq(&rna_bases)
    }
}

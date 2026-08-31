use crate::cai::tri2aa;

/// Conver CDS sequence to amino acids sequence
pub struct CDS2AA<'a> {
    /// Sequence name
    #[allow(dead_code)]
    pub seq_id: &'a String,
    /// Input cds sequence
    pub cds_seq: &'a String,
}

impl<'a> CDS2AA<'a> {
    pub fn new(seq_id: &'a String, cds_seq: &'a String) -> Self {
        CDS2AA { seq_id, cds_seq }
    }

    /// cds 2 aa
    pub fn to_aa(&self) -> String {
        let mut out: Vec<String> = Vec::new();
        for tri in self.cds_seq.chars().collect::<Vec<_>>().chunks(3) {
            let tri = String::from_iter(tri);
            out.push(tri2aa(&tri));
        }
        out.join("")
    }
}

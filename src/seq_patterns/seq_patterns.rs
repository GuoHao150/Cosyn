use crate::utils::{FaReader, SeqType};
use std::collections::HashSet;
use std::path::PathBuf;

/// Reader a fasta file and then store the avoid patterns
pub struct AvoidSeqs {
    /// Avoid seqs that should not be exsits in the optimized seq
    pub seqs: HashSet<String>,
    /// Avoid seqs with this number of homopolymers
    pub homopolymers_num: usize,
}

impl AvoidSeqs {
    pub fn new(path: Option<PathBuf>, homopolymers_num: usize) -> Self {
        match path {
            Some(p) => {
                let fa_reader = FaReader::new(p, SeqType::MOTIF);
                let mut motifs: HashSet<String> = HashSet::new();
                for (_h, motif) in fa_reader.id_seqs_table.into_iter() {
                    motifs.insert(AvoidSeqs::complementary_rev(&motif));
                    motifs.insert(motif);
                }
                AvoidSeqs {
                    seqs: motifs,
                    homopolymers_num,
                }
                .try_reduce_cds_pool()
            }
            None => AvoidSeqs {
                seqs: HashSet::new(),
                homopolymers_num,
            },
        }
    }

    fn complementary_rev(motif: &str) -> String {
        let mut out: Vec<String> = vec![];
        for c in motif.chars() {
            if c == 'A' {
                out.push("T".to_string());
            } else if c == 'T' {
                out.push("A".to_string());
            } else if c == 'C' {
                out.push("G".to_string());
            } else if c == 'G' {
                out.push("C".to_string());
            } else {
                panic!("The motif sequence that need to be avoid must only consists of `ATCG`");
            }
        }
        out.reverse();
        out.join("")
    }

    /// Check the motif sequences in the `self.seqs` with `filter_homopolymers`
    /// if return false then remove it
    fn try_reduce_cds_pool(self) -> Self {
        let h_num = self.homopolymers_num;
        AvoidSeqs {
            seqs: self
                .seqs
                .into_iter()
                .filter(|s| AvoidSeqs::_filter_homopolymers(h_num, s))
                .collect(),
            homopolymers_num: h_num,
        }
    }

    /// Check if the input sequence contains the avoid sequences
    /// if return true it means input seq not contain un-wanted motifs
    pub fn filter_cds(&self, in_seq: &str) -> bool {
        for motif in self.seqs.iter() {
            if in_seq.contains(motif) {
                return false;
            }
        }
        return true;
    }

    /// Get the unwanted motif sequences
    pub fn get_unwanted_seqs(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        for i in &self.seqs {
            out.insert(i.clone());
        }
        if self.homopolymers_num > 0 {
            ["A", "T", "C", "G"].into_iter().for_each(|n| {
                let mut v = Vec::with_capacity(self.homopolymers_num);
                for _ in 0..self.homopolymers_num {
                    v.push(n.to_string());
                }
                out.insert(v.join(""));
            })
        }
        out
    }

    /// Avoid homopolymers
    /// If return true, it means the input seq is good
    pub fn filter_homopolymers(&self, in_seq: &str) -> bool {
        if self.homopolymers_num == 0 {
            return true;
        }
        AvoidSeqs::_filter_homopolymers(self.homopolymers_num, in_seq)
    }

    fn _filter_homopolymers(homopolymers_num: usize, in_seq: &str) -> bool {
        let mut iter = in_seq.chars().into_iter();
        let mut cur_char = iter.next();
        let mut next_char = iter.next();
        let mut max_homo_size_in_seq: usize = 0;
        'outer: loop {
            if cur_char.is_none() || next_char.is_none() {
                break 'outer;
            }
            let cur_c = cur_char.unwrap();
            let next_c = next_char.unwrap();
            if cur_c != next_c {
                cur_char = next_char;
                next_char = iter.next();
                continue 'outer;
            }
            // elsee cur_c == next_c
            let mut cur_homo_size: usize = 2;
            let mut next_next_char = iter.next();
            'inner: loop {
                if next_next_char.is_none() {
                    break 'outer;
                }
                let nn_c = next_next_char.unwrap();
                if next_c == nn_c {
                    cur_homo_size += 1;
                    next_next_char = iter.next();
                    continue 'inner;
                } else {
                    if max_homo_size_in_seq < cur_homo_size {
                        max_homo_size_in_seq = cur_homo_size;
                    }
                    cur_char = next_next_char;
                    next_char = iter.next();
                    continue 'outer;
                }
            }
        }
        return max_homo_size_in_seq < homopolymers_num;
    }
}

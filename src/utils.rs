use crate::cds::CDSSeq;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;

/// The input sequences type in a fasta file
#[derive(PartialEq)]
pub(crate) enum SeqType {
    AA,
    DNA,
    RNA,
    MOTIF,
    /// Like DNA but without CDS validation (no multiple-of-3 or codon checks).
    /// Used for raw MFE calculation on arbitrary-length DNA/RNA sequences.
    RAW,
}

pub(crate) struct FaReader {
    fa_file: PathBuf,
    seq_header: VecDeque<String>,
    seq_type: SeqType,
    pub id_seqs_table: HashMap<String, String>,
}

impl FaReader {
    pub fn new(fa: PathBuf, seq_type: SeqType) -> Self {
        let mut out = FaReader {
            fa_file: fa.clone(),
            seq_header: VecDeque::new(),
            seq_type: seq_type,
            id_seqs_table: HashMap::new(),
        };
        out._read().err();
        out._check_aa_seqs().err();
        out
    }

    /// If the input sequences are `aa` but only contains `A` `T` `C` `G`
    /// then give some warnings   
    pub fn _check_aa_seqs(&mut self) -> io::Result<()> {
        if self.seq_type == SeqType::AA {
            for (idx, seq) in &self.id_seqs_table {
                if CDSSeq::check_base(&seq[..]) {
                    println!("Warning: You specified the input sequence as amino acids but the {} only contains `ATCG`", idx);
                    loop {
                        let mut user_input = String::new();
                        let stdin = io::stdin();
                        println!(
                            "Do you want to still process it as an amino acids sequences? (Y/N)"
                        );
                        stdin.read_line(&mut user_input).err();
                        match &user_input.trim()[..] {
                            "Y" | "y" => {
                                break;
                            }
                            "N" | "n" => {
                                std::process::exit(1);
                            }
                            _ => {
                                continue;
                            }
                        }
                    }
                }
            }
        }

        if self.seq_type == SeqType::DNA {
            for (idx, seq) in &self.id_seqs_table {
                if !CDSSeq::check_base(&seq[..]) {
                    println!(
                        "Error: The bases of input DNA sequence from {} was not just `ATCG`",
                        idx
                    );
                    std::process::exit(1);
                }
            }
        }

        Ok(())
    }

    /// Read the fasta file and store the data in a hash-map
    /// and check if the sequences in the fasta file are CDS sequences
    fn _read(&mut self) -> io::Result<()> {
        if !self.fa_file.exists() {
            println!("Input file {:?} not exists", self.fa_file.to_str());
            std::process::exit(1);
        }
        let fa_file = self.fa_file.to_str().unwrap().to_string();
        let file = File::open(fa_file).unwrap();
        let mut lines = io::BufReader::new(file).lines();

        let mut line = lines.next().unwrap()?;
        if !line.starts_with(">") {
            panic!("The first line of your input fasta file was not starts with > ");
        }
        'out: loop {
            if line.starts_with(">") {
                let header = line.clone().trim_end().to_owned();
                self.seq_header.push_back(header.clone());
                let mut seq = "".to_owned();
                let mut next_line = lines.next();
                'inner: loop {
                    match next_line {
                        None => {
                            self.id_seqs_table.insert(header.clone(), seq.clone());
                            break 'out;
                        }
                        Some(Ok(ref next_line_ref)) => {
                            if !next_line_ref.starts_with(">") {
                                seq.push_str(next_line_ref.trim_end());
                                next_line = lines.next();
                                continue 'inner;
                            } else {
                                self.id_seqs_table.insert(header.clone(), seq.clone());
                                line = next_line_ref.to_owned();
                                break 'inner;
                            }
                        }
                        _ => break 'out,
                    }
                }
            }
        }
        let mut upcase_seqs = HashMap::new();
        for (i, s) in &self.id_seqs_table {
            upcase_seqs.insert(i.to_owned(), s.to_uppercase());
        }
        self.id_seqs_table = upcase_seqs;
        // check the input CDS sequences
        for (idx, s) in &self.id_seqs_table {
            // not check for mofit sequences
            if self.seq_type == SeqType::MOTIF {
                let base_checked = CDSSeq::check_base(s);
                if !base_checked {
                    println!(
                        "The avoid sequence you provide is not a DNA sequence: {}",
                        idx
                    );
                    std::process::exit(1);
                }
                break;
            }
            if self.seq_type != SeqType::AA && self.seq_type != SeqType::RAW {
                let size_checked = CDSSeq::check_size(s);
                if !size_checked {
                    println!("The size of DNA sequence {} is not a multiple of 3, check your input DNA sequence", idx);
                    std::process::exit(1);
                }
                if self.seq_type == SeqType::DNA {
                    let base_checked = CDSSeq::check_base(s);
                    if !base_checked {
                        println!("The bases of DNA sequence {} is not `ATCG`", idx);
                        std::process::exit(1);
                    }
                    let triplet_checked = CDSSeq::check_triplets(s);
                    if !triplet_checked {
                        println!("The triplet in DNA sequence {} is illegal", idx);
                        std::process::exit(1);
                    }
                }
            }
            // For RAW type: only validate bases (ATCG), no CDS checks
            if self.seq_type == SeqType::RAW {
                let base_checked = CDSSeq::check_base(s);
                if !base_checked {
                    println!("The bases of DNA sequence {} is not `ATCG`", idx);
                    std::process::exit(1);
                }
            }
        }
        Ok(())
    }
}

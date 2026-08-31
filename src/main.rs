mod beam_search;
mod cai;
mod cds;
mod codon_pll_binding;
mod gc;
mod mcmc;
mod mfe_binding;
mod mutate2stop;
mod palindrome;
mod seq_convertor;
mod seq_patterns;
mod utils;

use beam_search::{BeamSearch, CodonTables, OptimizeObject, SearchConfig, SnpMutation};
use cai::{try_aa_to_idx, CaiMode, RawCaiCalculator};
use cds::CDSSeq;
use codon_pll_binding::{CodonPllCalculator, MAX_PLL_AA_LEN, MAX_PLL_CODON_COUNT};
use gc::GcCalculator;
use mcmc::run_weak_head_mcmc;
use mcmc::MCMCSearch;
use mfe_binding::{set_mfe_max_pair_dist, MfeCalculator};

fn configure_mfe_window(window: usize) {
    set_mfe_max_pair_dist(if window == 0 { None } else { Some(window) });
}
use mutate2stop::M2sCalculator;
use num_cpus;
use palindrome::PalinCalculator;
use regex::Regex;
use seq_convertor::{AA2CDS, CDS2AA, RNA2CDS};
use seq_patterns::AvoidSeqs;
use serde_json::{json, to_string_pretty, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::Arc;
use structopt::StructOpt;
use threadpool::ThreadPool;
use utils::FaReader;
use utils::SeqType;
mod cli;
use cli::*;
mod config;
use config::ConfigFile;
mod explore;
use explore::run_explore;

/// A private function to save the json file
fn save_json_content(
    content: Value,
    prefix: Option<String>,
    outdir: PathBuf,
    lambda: Option<f64>,
    suffix: &str,
    output_fasta: bool,
) -> std::io::Result<()> {
    let content_fa = content.clone();
    let content = to_string_pretty(&content)?;
    let out_file;
    let out_fa;
    if prefix.is_none() {
        if lambda.is_some() {
            out_file =
                outdir
                    .clone()
                    .join(format!("out_lambda_{}_{}.json", lambda.unwrap(), suffix));
            out_fa =
                outdir
                    .clone()
                    .join(format!("out_lambda_{}_{}.fasta", lambda.unwrap(), suffix));
        } else {
            out_file = outdir.clone().join(format!("out_{}.json", suffix));
            out_fa = outdir.clone().join(format!("out_{}.fasta", suffix));
        }
    } else {
        let prefix = prefix.clone().unwrap();
        if lambda.is_some() {
            out_file = outdir.clone().join(format!(
                "{}_out_lambda_{}_{}.json",
                prefix,
                lambda.unwrap(),
                suffix
            ));
            out_fa = outdir.clone().join(format!(
                "{}_out_lambda_{}_{}.fasta",
                prefix,
                lambda.unwrap(),
                suffix
            ));
        } else {
            out_file = outdir
                .clone()
                .join(format!("{}_out_{}.json", prefix, suffix));
            out_fa = outdir
                .clone()
                .join(format!("{}_out_{}.fasta", prefix, suffix));
        }
    }
    let fw_handle = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_file)?;
    let mut fw = BufWriter::new(&fw_handle);
    fw.write_all(content.as_bytes())?;
    if output_fasta {
        let fw_handle_fa = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(out_fa)?;
        let mut fw_fa = BufWriter::new(&fw_handle_fa);
        if let Some(data) = content_fa["results"].as_array() {
            let mut idx = 0;
            for s in data {
                let gc = s["GC%"].as_f64().unwrap_or(0.0);
                let mfe = s["mfe"].as_f64().unwrap_or(0.0);
                let cai = s["raw_cai"].as_f64().unwrap_or(0.0);
                let ac = s["arithmetic_cai"].as_f64().unwrap_or(0.0);
                let cds = s["optimized_cds"].as_str().unwrap_or("").to_string();
                let seq_id = s["seq_id"].as_str().unwrap().to_string();
                let palindrome_score = s["palindrome_score"].as_f64().unwrap_or(0.0);
                let m2s_score = s["m2s_score"].as_f64().unwrap_or(0.0);
                let pll_score = s["raw_pll"].as_f64().unwrap_or(0.0);
                let header = format!(
                    "{} index: {} GC: {} mfe: {} cai: {} parlindrom_score: {} mutate2stop_score: {} arithmetic_cai: {} pll_score: {}\n",
                    seq_id, idx, gc, mfe, cai, palindrome_score, m2s_score, ac, pll_score
                );
                let seq = format!("{}\n", cds);
                fw_fa.write_all(header.as_bytes())?;
                fw_fa.write_all(seq.as_bytes())?;
                idx += 1;
            }
        }
    }
    Ok(())
}
const LEGAL_SHORTS: &[&str; 88] = &[
    "bcl", "bsc", "ems", "bhp", "bfc", "csto", "cpla", "lu", "th", "cpit", "ac", "v", "bna",
    "ctes", "cspl", "ct", "hlv", "bsn", "cthy", "ceso", "bam", "cliv", "li", "cgal", "sit", "chea",
    "msg", "csal", "pr", "cski", "ccol", "cbre", "cart", "cmus", "pi", "ms", "bce", "as", "egj",
    "cfal", "wb", "ema", "ty", "av", "cvag", "bch", "cec", "cs", "bhi", "cner", "s", "bp", "bac",
    "kc", "cute", "csmo", "ckid", "cton", "cduo", "cen", "cova", "cbra", "clun", "cpro", "ag",
    "cfat", "hc", "cadr", "curi", "sp", "ov", "ne", "csma", "ft", "at", "pa", "u", "haa", "sns",
    "b", "aa", "capp", "cpan", "crec", "bc", "clym", "sse", "ecoli",
];
fn check_table_short(name: &str) -> bool {
    // Accept built-in short names or .toml file paths for custom tables.
    if CodonTables::is_toml_path(name) {
        return true;
    }
    LEGAL_SHORTS.iter().any(|x| (*x).eq(name))
}

/// If `name` is a .toml path, load the custom codon table from it.
/// Returns an error message on failure, or Ok(()) on success.
fn load_codon_table(name: &str) -> Result<(), String> {
    if CodonTables::is_toml_path(name) {
        CodonTables::load_custom_table(name)
    } else {
        Ok(())
    }
}

fn parse_aa_mutation(mutation_site: &str, aa_seq: &str) -> (String, usize, String) {
    let aa_size = aa_seq.len();
    let re = Regex::new(r"(?P<number>\d+)|(?P<word>[a-zA-Z]+)").unwrap();
    let mut tmp_pos: Vec<u32> = Vec::new();
    let mut tmp_word: Vec<String> = Vec::new();
    for caps in re.captures_iter(&mutation_site[..]) {
        if let Some(number) = caps.name("number") {
            tmp_pos.push(number.as_str().parse::<u32>().unwrap());
        }
        if let Some(word) = caps.name("word") {
            tmp_word.push(word.as_str().to_string());
        }
    }
    if (tmp_pos.len() != 1) | (tmp_word.len() != 2) {
        println!(
            "Error: the format of input {} was wrong. It should be something like `A20R`",
            mutation_site
        );
        std::process::exit(1);
    }

    let mutate_pos = tmp_pos[0] as usize;
    if (mutate_pos == 0) | (mutate_pos > aa_size) {
        println!(
            "Error: the input mutation positon should be within [1, {}] but you provided {}",
            aa_size, mutate_pos
        );
    }

    let aa_in_seq = aa_seq[mutate_pos - 1..mutate_pos].to_string();
    if tmp_word[0].ne(&aa_in_seq) {
        println!("Error: The amino acid in the amino acids sequence is {} but in you mutation input the {} position is {}",
            aa_in_seq, mutate_pos, tmp_word[0].clone());
        std::process::exit(1);
    }
    for aa in &tmp_word {
        if try_aa_to_idx(&aa[..]).is_none() {
            println!(
                "Amino acid {} from input {} was not a standard one word character for amino acid",
                aa, mutation_site
            );
            std::process::exit(1);
        }
    }

    (tmp_word[0].clone(), mutate_pos, tmp_word[1].clone())
}

/// Parse `--mcmc-region` string of the form `start-end` (base positions).
/// Returns a vector of 0-based codon indices that lie inside the region
/// [start, end). Validates that positions are multiples of 3, within the
/// CDS length, and that the input is not amino-acid sequence.
fn parse_mcmc_region(
    region_str: &Option<String>,
    cds_len: usize,
    is_aa: bool,
) -> Option<Vec<usize>> {
    let s = region_str.as_ref()?;
    if is_aa {
        println!("Error: --mcmc-region requires DNA/RNA input, not amino acid sequences");
        std::process::exit(1);
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        println!("Error: --mcmc-region must be in format start-end (e.g. 30-90)");
        std::process::exit(1);
    }
    let parse_pos = |label: &str, value: &str| -> usize {
        value.parse::<usize>().unwrap_or_else(|_| {
            println!("Error: --mcmc-region {} must be an integer", label);
            std::process::exit(1);
        })
    };
    let start = parse_pos("start", parts[0]);
    let end = parse_pos("end", parts[1]);

    if start % 3 != 0 || end % 3 != 0 {
        println!(
            "Error: --mcmc-region start ({}) and end ({}) must be multiples of 3",
            start, end
        );
        std::process::exit(1);
    }
    if end > cds_len {
        println!(
            "Error: --mcmc-region end ({}) exceeds CDS length ({})",
            end, cds_len
        );
        std::process::exit(1);
    }
    if start >= end {
        println!(
            "Error: --mcmc-region start ({}) must be strictly less than end ({})",
            start, end
        );
        std::process::exit(1);
    }

    Some((start / 3..end / 3).collect())
}

fn run_fit(opt: &FitOptions) -> std::io::Result<()> {
    if opt.lambda < 0.0 {
        println!("The lambda paramter should greater than 0");
        std::process::exit(1);
    }
    if opt.is_aa_seq && opt.is_rna {
        println!("The input fasta file can't be amino acids and rna at the same time");
        std::process::exit(1);
    }
    if opt.seq_parallel == 0 {
        println!("The number of simultaneously process sequences should greater or equal to 1");
        std::process::exit(1);
    }
    if !check_table_short(&opt.freq_table_info) {
        println!("The provided short name must be one of {:?}", LEGAL_SHORTS);
        std::process::exit(1);
    }
    if opt.homopolymers < 6 {
        println!("Warning: the maximum size of homopolymers you set is less than 7");
    }
    let seq_type = if opt.is_aa_seq {
        SeqType::AA
    } else if opt.is_rna {
        SeqType::RNA
    } else {
        SeqType::DNA
    };
    let threads = if opt.threads.is_none() {
        num_cpus::get()
    } else {
        opt.threads.unwrap() as usize
    };

    let mfe_method = opt.mfe_method;
    configure_mfe_window(opt.mfe_window);
    let no_verbose = opt.no_verbose;
    let if_maximize_loss = opt.maximize_loss;
    let optimize_palindrome = opt.palindrome;

    // Shared inner thread pool for beam search loss computation.
    // Creating a single pool avoids N×threads total workers when
    // processing multiple sequences with seq_parallel > 1.
    let shared_inner_pool = Arc::new(ThreadPool::new(threads));
    let trace_pool = Arc::new(ThreadPool::new(num_cpus::get()));

    let use_cmgmp = opt.cmgmp;
    let use_pmgc = opt.pmgc;
    let use_gc_cai = opt.gc_cai;
    let use_mfe_cai = opt.mfe_cai;

    let tmp_flags = [
        use_cmgmp as u8,
        use_pmgc as u8,
        use_gc_cai as u8,
        use_mfe_cai as u8,
    ];
    let flag_sum = tmp_flags.iter().sum::<u8>();
    if flag_sum > 1 {
        println!("Can't use more than one optimization method at the same time");
        std::process::exit(1);
    }

    if let Err(e) = load_codon_table(&opt.freq_table_info) {
        println!("Error loading custom codon table: {}", e);
        std::process::exit(1);
    }
    let codon_table = CodonTables::short2enum(&opt.freq_table_info.clone());
    let use_mcmc = opt.search_method.to_lowercase() == "mcmc";
    let mut beam_searchers: HashMap<String, BeamSearch> = HashMap::new();
    let mut mcmc_searchers: HashMap<String, MCMCSearch> = HashMap::new();
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    let avoid_seqs = Arc::new(AvoidSeqs::new(opt.avoid_seqs.clone(), opt.homopolymers));
    let mut bar_position = 0;
    let weight_gc = opt.weight_gc;
    let weight_cai = opt.weight_cai;
    let weight_palindrome = opt.weight_palindrome;
    let weight_m2s = opt.weight_m2s;
    let process_pool = ThreadPool::new(opt.seq_parallel as usize);
    let (tx, rx) = channel();
    let mut weak_head_jobs: usize = 0;

    for (idx, s) in &fa_reader.id_seqs_table {
        let (seq, aa_str) = if opt.is_aa_seq {
            let aa_c = AA2CDS::new(idx, s);
            (aa_c.to_cds(), s.clone())
        } else if use_mcmc && opt.is_rna {
            // For MCMC on RNA, preserve the original transcribed CDS
            // so that regions outside --mcmc-region stay unchanged.
            let rna_c = RNA2CDS::new(idx, s);
            let cds = rna_c.to_cds();
            let aa_seq = CDS2AA::new(idx, &cds).to_aa();
            (cds, aa_seq)
        } else if use_mcmc {
            // For MCMC on DNA, preserve the original CDS.
            let aa_seq = CDS2AA::new(idx, s).to_aa();
            (s.clone(), aa_seq)
        } else if opt.is_rna {
            let rna_c = RNA2CDS::new(idx, s);
            let cds = rna_c.to_cds();
            let cds2aa = CDS2AA::new(idx, &cds);
            let aa_seq = cds2aa.to_aa();
            let aa2cds = AA2CDS::new(idx, &aa_seq);
            (aa2cds.to_cds(), aa_seq)
        } else {
            // Beam search: randomize initial CDS via synonymous codon sampling
            let cds2aa = CDS2AA::new(idx, s);
            let aa_seq = cds2aa.to_aa();
            let aa2cds = AA2CDS::new(idx, &aa_seq);
            (aa2cds.to_cds(), aa_seq)
        };
        let aa_seq_vec: Vec<String> = aa_str.chars().map(|c| c.to_string()).collect();
        let opt_obj = if use_gc_cai {
            if optimize_palindrome {
                OptimizeObject::GcCaiPalin
            } else {
                OptimizeObject::GcCai
            }
        } else if use_mfe_cai {
            if optimize_palindrome {
                OptimizeObject::MfeCaiPalin
            } else {
                OptimizeObject::MfeCai
            }
        } else if use_cmgmp {
            OptimizeObject::MfeCaiGcM2sPalin
        } else if use_pmgc {
            OptimizeObject::CaiGcM2sPalin
        } else {
            if optimize_palindrome {
                OptimizeObject::MfeGcCaiPalin
            } else {
                OptimizeObject::MfeGcCai // default mfe/gc/gai
            }
        };

        let config = SearchConfig {
            lambda: opt.lambda,
            mfe_method: mfe_method.clone(),
            weight_gc,
            weight_cai,
            weight_palindrome,
            weight_m2s,
            weight_pll: 0.0,
            model_path: None,
            if_minimize_loss: !if_maximize_loss,
            opt_obj,
            cai_mode: opt.cai_mode,
            mfe_pll_start_codon: opt.mfe_pll_start_codon,
            weak_head_bases: if use_mcmc { 0 } else { opt.weak_head },
        };

        if use_mcmc && opt.weak_head > 0 {
            // Two-phase weak-head MCMC: run directly via process_pool.
            // Phase 1 optimizes the 5'-end head for unstable structure;
            // Phase 2 optimizes the tail with user's original parameters;
            // then concatenate and evaluate full metrics.
            let sender = tx.clone();
            let seq_id = idx.clone();
            let ct = codon_table.clone();
            let cfg = config.clone();
            let aa = aa_seq_vec.clone();
            let cds = seq.clone();
            let av = Arc::clone(&avoid_seqs);
            let iters = opt.mcmc_iterations;
            let muts = opt.mcmc_mutations;
            let acc = opt.mcmc_accept_prob;
            let temp = opt.mcmc_temperature;
            let tr = opt.mcmc_traces;
            let wh = opt.weak_head;
            let mf = mfe_method.clone();
            let nv = no_verbose;
            let bp = bar_position;

            process_pool.execute(move || {
                let results = run_weak_head_mcmc(
                    &seq_id, &aa, &cds, &ct, &cfg, av, iters, muts, acc, temp, tr, wh, &mf, bp, nv,
                );
                sender
                    .send(results)
                    .expect("failed to send weak-head MCMC results");
            });
            weak_head_jobs += 1;
        } else if use_mcmc {
            let mutable_positions = parse_mcmc_region(&opt.mcmc_region, seq.len(), opt.is_aa_seq);
            let searcher = MCMCSearch::new(
                idx.clone(),
                aa_seq_vec,
                seq,
                codon_table.clone(),
                config,
                Arc::clone(&avoid_seqs),
                opt.mcmc_iterations,
                opt.mcmc_mutations,
                opt.mcmc_accept_prob,
                opt.mcmc_temperature,
                mutable_positions,
                no_verbose,
                bar_position,
                opt.mcmc_traces,
                Arc::clone(&trace_pool),
            );
            mcmc_searchers.insert(idx.clone(), searcher);
        } else {
            let bs = BeamSearch::new(
                CDSSeq::new(seq),
                opt.win_size,
                opt.rnd_size,
                opt.num_outputs,
                idx.clone(),
                Arc::clone(&avoid_seqs),
                codon_table.clone(),
                bar_position,
                config,
                Arc::clone(&shared_inner_pool),
                no_verbose,
            );
            beam_searchers.insert(idx.clone(), bs);
        }
        bar_position += 1;
    }

    let n_jobs = weak_head_jobs
        + if use_mcmc {
            mcmc_searchers.len()
        } else {
            beam_searchers.len()
        };

    let mod_num = opt.seq_parallel as i32;
    if use_mcmc {
        for (idx, (seq_id, searched)) in mcmc_searchers.into_iter().enumerate() {
            let sender = tx.clone();
            process_pool.execute(move || {
                let new_bar_pos = (idx as i32 % mod_num) as u16;
                sender
                    .send(searched.set_bar_position(new_bar_pos).search())
                    .expect(&format!("failed to send searched results of {}", seq_id));
            })
        }
    } else {
        for (idx, (seq_id, searched)) in beam_searchers.into_iter().enumerate() {
            let sender = tx.clone();
            process_pool.execute(move || {
                let new_bar_pos = (idx as i32 % mod_num) as u16;
                sender
                    .send(searched.set_bar_position(new_bar_pos).search())
                    .expect(&format!("failed to send searched results of {}", seq_id));
            })
        }
    }

    for idx in 0..n_jobs {
        let out = rx
            .recv()
            .unwrap()
            .into_iter()
            .map(|sp| sp.to_json())
            .collect::<Vec<_>>();
        let mut content = json!({
            "results": out,
        });
        if let Some(obj) = content.as_object_mut() {
            obj.insert(
                "params".to_string(),
                serde_json::to_value(opt)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
            );
        }
        let suffix = format!("seq_{}_fit", idx);
        save_json_content(
            content,
            opt.prefix.clone(),
            opt.outdir.clone(),
            Some(opt.lambda),
            &suffix,
            true,
        )?;
    }
    Ok(())
}

fn run_ufit(opt: &UfitOptions) -> std::io::Result<()> {
    if opt.is_aa_seq && opt.is_rna {
        println!("The input fasta file can't be amino acids and rna at the same time");
        std::process::exit(1);
    }
    if opt.seq_parallel == 0 {
        println!("The number of simultaneously process sequences should greater or equal to 1");
        std::process::exit(1);
    }
    if !check_table_short(&opt.freq_table_info) {
        println!("The provided short name must be one of {:?}", LEGAL_SHORTS);
        std::process::exit(1);
    }
    if opt.homopolymers < 6 {
        println!("Warning: the maximum size of homopolymers you set is less than 7");
    }
    let seq_type = if opt.is_aa_seq {
        SeqType::AA
    } else if opt.is_rna {
        SeqType::RNA
    } else {
        SeqType::DNA
    };
    let threads = if opt.threads.is_none() {
        num_cpus::get()
    } else {
        opt.threads.unwrap() as usize
    };

    // When PLL is enabled the inner thread pool feeds GPU batches.
    // A larger pool keeps the GPU busy; default to at least 32 workers.
    let inner_threads = if opt.pll_threads > 0 {
        opt.pll_threads
    } else if opt.weight_pll != 0.0 && opt.model_path.is_some() {
        threads.max(32)
    } else {
        threads
    };

    // Configure GPU batch accumulation before the first PLL call.
    if opt.model_path.is_some() {
        codon_pll_binding::call_codon_pll::configure_pll_batch_params(
            opt.pll_batch_size,
            opt.pll_timeout_ms,
        );
    }

    let mfe_method = opt.mfe_method;
    configure_mfe_window(opt.mfe_window);
    let no_verbose = opt.no_verbose;
    let if_maximize_loss = opt.maximize_loss;

    // Shared inner thread pool (see run_fit for rationale).
    let shared_inner_pool = Arc::new(ThreadPool::new(inner_threads));
    let trace_pool = Arc::new(ThreadPool::new(num_cpus::get()));

    if let Err(e) = load_codon_table(&opt.freq_table_info) {
        println!("Error loading custom codon table: {}", e);
        std::process::exit(1);
    }
    let codon_table = CodonTables::short2enum(&opt.freq_table_info.clone());
    let use_mcmc = opt.search_method.to_lowercase() == "mcmc";
    let mut beam_searchers: HashMap<String, BeamSearch> = HashMap::new();
    let mut mcmc_searchers: HashMap<String, MCMCSearch> = HashMap::new();
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    let avoid_seqs = Arc::new(AvoidSeqs::new(opt.avoid_seqs.clone(), opt.homopolymers));
    let mut bar_position = 0;
    let weight_gc = opt.weight_gc;
    let weight_cai = opt.weight_cai;
    let weight_palindrome = opt.weight_palindrome;
    let weight_m2s = opt.weight_m2s;
    let weight_mfe = opt.weight_mfe;
    let weight_pll = opt.weight_pll;
    let model_path = opt
        .model_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let opt_obj = OptimizeObject::UnifiedObj;
    let process_pool = ThreadPool::new(opt.seq_parallel as usize);
    let (tx, rx) = channel();
    let mut weak_head_jobs: usize = 0;

    for (idx, s) in &fa_reader.id_seqs_table {
        let (seq, aa_str) = if opt.is_aa_seq {
            let aa_c = AA2CDS::new(idx, s);
            (aa_c.to_cds(), s.clone())
        } else if use_mcmc && opt.is_rna {
            // For MCMC on RNA, preserve the original transcribed CDS.
            let rna_c = RNA2CDS::new(idx, s);
            let cds = rna_c.to_cds();
            let aa_seq = CDS2AA::new(idx, &cds).to_aa();
            (cds, aa_seq)
        } else if use_mcmc {
            // For MCMC on DNA, preserve the original CDS.
            let aa_seq = CDS2AA::new(idx, s).to_aa();
            (s.clone(), aa_seq)
        } else if opt.is_rna {
            let rna_c = RNA2CDS::new(idx, s);
            let cds = rna_c.to_cds();
            let cds2aa = CDS2AA::new(idx, &cds);
            let aa_seq = cds2aa.to_aa();
            let aa2cds = AA2CDS::new(idx, &aa_seq);
            (aa2cds.to_cds(), aa_seq)
        } else {
            // Beam search: randomize initial CDS via synonymous codon sampling
            let cds2aa = CDS2AA::new(idx, s);
            let aa_seq = cds2aa.to_aa();
            let aa2cds = AA2CDS::new(idx, &aa_seq);
            (aa2cds.to_cds(), aa_seq)
        };
        let aa_seq_vec: Vec<String> = aa_str.chars().map(|c| c.to_string()).collect();

        if weight_pll != 0.0 && model_path.is_some() {
            let codon_count = seq.len() / 3;
            if codon_count > MAX_PLL_CODON_COUNT {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "[codon_pll] Sequence '{}' has {} codons, exceeding the PLL limit of {} codons (~{} amino acids with stop codon).",
                        idx.trim_start_matches('>'), codon_count, MAX_PLL_CODON_COUNT, MAX_PLL_AA_LEN
                    ),
                ));
            }
        }

        let config = SearchConfig {
            lambda: opt.lambda,
            mfe_method: mfe_method.clone(),
            weight_gc,
            weight_cai,
            weight_palindrome,
            weight_m2s,
            weight_pll,
            model_path: model_path.clone(),
            if_minimize_loss: !if_maximize_loss,
            opt_obj: opt_obj.clone(),
            cai_mode: opt.cai_mode,
            mfe_pll_start_codon: opt.mfe_pll_start_codon,
            weak_head_bases: if use_mcmc { 0 } else { opt.weak_head },
        };

        if use_mcmc && opt.weak_head > 0 {
            // Two-phase weak-head MCMC: run directly via process_pool.
            let sender = tx.clone();
            let seq_id = idx.clone();
            let ct = codon_table.clone();
            let cfg = config.clone();
            let aa = aa_seq_vec.clone();
            let cds = seq.clone();
            let av = Arc::clone(&avoid_seqs);
            let iters = opt.mcmc_iterations;
            let muts = opt.mcmc_mutations;
            let acc = opt.mcmc_accept_prob;
            let temp = opt.mcmc_temperature;
            let tr = opt.mcmc_traces;
            let wh = opt.weak_head;
            let mf = mfe_method.clone();
            let nv = no_verbose;
            let bp = bar_position;

            process_pool.execute(move || {
                let results = run_weak_head_mcmc(
                    &seq_id, &aa, &cds, &ct, &cfg, av, iters, muts, acc, temp, tr, wh, &mf, bp, nv,
                );
                sender
                    .send(results)
                    .expect("failed to send weak-head MCMC results");
            });
            weak_head_jobs += 1;
        } else if use_mcmc {
            let mutable_positions = parse_mcmc_region(&opt.mcmc_region, seq.len(), opt.is_aa_seq);
            let searcher = MCMCSearch::new(
                idx.clone(),
                aa_seq_vec,
                seq,
                codon_table.clone(),
                config,
                Arc::clone(&avoid_seqs),
                opt.mcmc_iterations,
                opt.mcmc_mutations,
                opt.mcmc_accept_prob,
                opt.mcmc_temperature,
                mutable_positions,
                no_verbose,
                bar_position,
                opt.mcmc_traces,
                Arc::clone(&trace_pool),
            );
            mcmc_searchers.insert(idx.clone(), searcher);
        } else {
            let mut bs = BeamSearch::new(
                CDSSeq::new(seq),
                opt.win_size,
                opt.rnd_size,
                opt.num_outputs,
                idx.clone(),
                Arc::clone(&avoid_seqs),
                codon_table.clone(),
                bar_position,
                config,
                Arc::clone(&shared_inner_pool),
                no_verbose,
            );
            bs.weight_mfe = weight_mfe;
            beam_searchers.insert(idx.clone(), bs);
        }
        bar_position += 1;
    }
    let n_jobs = weak_head_jobs
        + if use_mcmc {
            mcmc_searchers.len()
        } else {
            beam_searchers.len()
        };
    let mod_num = opt.seq_parallel as i32;
    if use_mcmc {
        for (idx, (seq_id, searched)) in mcmc_searchers.into_iter().enumerate() {
            let sender = tx.clone();
            process_pool.execute(move || {
                let new_bar_pos = (idx as i32 % mod_num) as u16;
                sender
                    .send(searched.set_bar_position(new_bar_pos).search())
                    .expect(&format!("failed to send searched results of {}", seq_id));
            })
        }
    } else {
        for (idx, (seq_id, searched)) in beam_searchers.into_iter().enumerate() {
            let sender = tx.clone();
            process_pool.execute(move || {
                let new_bar_pos = (idx as i32 % mod_num) as u16;
                sender
                    .send(searched.set_bar_position(new_bar_pos).search())
                    .expect(&format!("failed to send searched results of {}", seq_id));
            })
        }
    }

    for idx in 0..n_jobs {
        let out = rx
            .recv()
            .unwrap()
            .into_iter()
            .map(|sp| sp.to_json())
            .collect::<Vec<_>>();
        let mut content = json!({
            "results": out,
        });
        if let Some(obj) = content.as_object_mut() {
            obj.insert(
                "params".to_string(),
                serde_json::to_value(opt)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
            );
        }
        let suffix = format!("seq_{}_fit", idx);
        save_json_content(
            content,
            opt.prefix.clone(),
            opt.outdir.clone(),
            Some(opt.lambda),
            &suffix,
            true,
        )?;
    }
    Ok(())
}

fn run_cai(opt: &CaiOptions) -> std::io::Result<()> {
    let seq_type = SeqType::DNA;
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    let mut caculators = Vec::new();
    if !check_table_short(&opt.freq_table_info) {
        println!("The provided short name must be one of {:?}", LEGAL_SHORTS);
        std::process::exit(0);
    }
    if let Err(e) = load_codon_table(&opt.freq_table_info) {
        println!("Error loading custom codon table: {}", e);
        std::process::exit(1);
    }
    let codon_table = CodonTables::short2enum(&opt.freq_table_info.clone());
    for (idx, seq) in &fa_reader.id_seqs_table {
        caculators.push(RawCaiCalculator::new(idx, seq, codon_table.clone()));
    }
    let cai_outs = caculators
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();
    let content = json!({
        "results": cai_outs,
    });
    let suffix = "cai";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn run_mfe(opt: &MfeOptions) -> std::io::Result<()> {
    let seq_type = SeqType::RAW;
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    configure_mfe_window(opt.mfe_window);

    // Fast path: a single sequence does not benefit from the thread pool overhead.
    let mfe_outs = if fa_reader.id_seqs_table.len() <= 1 {
        fa_reader
            .id_seqs_table
            .iter()
            .map(|(idx, seq)| MfeCalculator::new(idx, seq, opt.mfe_method).to_json())
            .collect::<Vec<_>>()
    } else {
        let pool = ThreadPool::new(opt.threads.unwrap_or_else(|| num_cpus::get() as u32) as usize);
        let (tx, rx) = channel();

        for (idx, seq) in &fa_reader.id_seqs_table {
            let idx = idx.clone();
            let seq = seq.clone();
            let mfe_method = opt.mfe_method;
            let sender = tx.clone();
            pool.execute(move || {
                let calc = MfeCalculator::new(&idx, &seq, mfe_method);
                sender
                    .send(calc.to_json())
                    .expect("failed to send MFE result");
            });
        }

        let mut outs = Vec::with_capacity(fa_reader.id_seqs_table.len());
        for _ in 0..fa_reader.id_seqs_table.len() {
            outs.push(rx.recv().expect("failed to receive MFE result"));
        }
        outs
    };

    let content = json!({
        "results": mfe_outs,
    });
    let suffix = "mfe";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn run_gc(opt: &GcOptions) -> std::io::Result<()> {
    let seq_type = SeqType::DNA;
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    let mut calculators = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        calculators.push(GcCalculator::new(idx, seq));
    }
    let gc_outs = calculators
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();
    let content = json!({
        "results": gc_outs,
    });
    let suffix = "gc";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn run_palindrome(opt: &PalindromeOptions) -> std::io::Result<()> {
    let seq_type = SeqType::DNA;
    let mut fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    for (_idx, dna) in &mut fa_reader.id_seqs_table {
        *dna = dna.replace("T", "U");
    }
    let mut calculators = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        calculators.push(PalinCalculator::new(idx, seq));
    }
    let palin_outs = calculators
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();
    let content = json!({
        "results": palin_outs,
    });
    let suffix = "palindrome";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn run_mutate2stop(opt: &M2sOptions) -> std::io::Result<()> {
    let fa_reader = FaReader::new(opt.fasta.clone(), SeqType::DNA);
    let mut calculators = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        calculators.push(M2sCalculator::new(idx, seq));
    }
    let m2s_outs = calculators
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();
    let content = json!({
        "results": m2s_outs,
    });
    let suffix = "mutate2stop";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn run_snp(opt: &SnpOptions) -> std::io::Result<()> {
    let use_cmgmp = opt.cmgmp;
    let use_pmgc = opt.pmgc;
    let use_gc_cai = opt.gc_cai;
    let use_mfe_cai = opt.mfe_cai;
    let tmp_flags = vec![
        use_cmgmp as u8,
        use_pmgc as u8,
        use_gc_cai as u8,
        use_mfe_cai as u8,
    ];
    let flag_sum = tmp_flags.iter().sum::<u8>();
    if flag_sum > 1 {
        println!("Can't use more than one optimization method at the same time");
        std::process::exit(1);
    }

    if let Err(e) = load_codon_table(&opt.freq_table_info) {
        println!("Error loading custom codon table: {}", e);
        std::process::exit(1);
    }
    let fa_reader = FaReader::new(opt.fasta.clone(), SeqType::DNA);
    let codon_table = CodonTables::short2enum(&opt.freq_table_info.clone());
    let use_mcmc = opt.search_method.to_lowercase() == "mcmc";
    let threads = if opt.threads.is_none() {
        num_cpus::get()
    } else {
        opt.threads.clone().unwrap() as usize
    };
    let optimize_palindrome = opt.palindrome;
    let if_maximize_loss = opt.maximize_loss;
    let no_verbose = opt.no_verbose;
    let weight_gc = opt.weight_gc;
    let weight_cai = opt.weight_cai;
    let weight_palindrome = opt.weight_palindrome;
    let weight_m2s = opt.weight_m2s;
    let mfe_method = opt.mfe_method;
    configure_mfe_window(opt.mfe_window);
    let avoid_seqs = Arc::new(AvoidSeqs::new(opt.avoid_seqs.clone(), opt.homopolymers));
    if fa_reader.id_seqs_table.len() != 1 {
        println!("Error: for now the SNP optimizing can only handle one CDS sequence");
        std::process::exit(1);
    }
    if let Some((idx, cds_seq)) = fa_reader.id_seqs_table.iter().next() {
        let (idx, cds_seq) = (idx.clone(), cds_seq.clone());
        // check the input cds_seq contains any `avoid seq` or `homopolymers`
        if (!avoid_seqs.filter_cds(&cds_seq)) | (!avoid_seqs.filter_homopolymers(&cds_seq)) {
            println!(
                "Error: input the sequence {} contains unwanted motif or homopolymers",
                idx
            );
            std::process::exit(1);
        }
        let aa_seq = CDS2AA::new(&idx, &cds_seq).to_aa();
        let aa_seq_vec: Vec<String> = aa_seq.chars().map(|c| c.to_string()).collect();
        let thread_pool = ThreadPool::new(threads);
        let opt_obj = if opt.gc_cai {
            if optimize_palindrome {
                OptimizeObject::GcCaiPalin
            } else {
                OptimizeObject::GcCai
            }
        } else if opt.mfe_cai {
            if optimize_palindrome {
                OptimizeObject::MfeCaiPalin
            } else {
                OptimizeObject::MfeCai
            }
        } else {
            if optimize_palindrome {
                OptimizeObject::MfeGcCaiPalin
            } else {
                OptimizeObject::MfeGcCai // default mfe/gc/gai
            }
        };
        let mutation_aa_sites = opt.mutations.clone();
        let mut mutations_pos = Vec::new();
        let mut target_aa = Vec::new();
        for mutation_site in &mutation_aa_sites {
            let (_f_aa, pos_aa, t_aa) = parse_aa_mutation(mutation_site, &aa_seq);
            if mutations_pos.contains(&pos_aa) {
                println!(
                    "Error: mutation postion {} have been provided repeatly",
                    pos_aa
                );
                std::process::exit(1);
            }
            mutations_pos.push(pos_aa);
            target_aa.push(t_aa.clone());
        }
        let config = SearchConfig {
            lambda: opt.lambda,
            mfe_method,
            weight_gc,
            weight_cai,
            weight_palindrome,
            weight_m2s,
            weight_pll: 0.0,
            model_path: None,
            if_minimize_loss: !if_maximize_loss,
            opt_obj: opt_obj.clone(),
            cai_mode: CaiMode::Scaled,
            mfe_pll_start_codon: opt.mfe_pll_start_codon,
            weak_head_bases: 0,
        };

        let best_sp = if use_mcmc {
            let trace_pool = Arc::new(ThreadPool::new(num_cpus::get()));
            // Build initial mutated CDS by applying the requested
            // amino-acid changes with random synonymous codons.
            let mut initial_cds_chars: Vec<char> = cds_seq.chars().collect();
            let mut rng = rand::thread_rng();
            for (&pos_1based, t_aa) in mutations_pos.iter().zip(target_aa.iter()) {
                let pos_0based = pos_1based - 1;
                if let Some(new_codon) =
                    mcmc::sample_synonymous_codon(t_aa.as_str(), &codon_table, &mut rng)
                {
                    let start = pos_0based * 3;
                    if start + 3 <= initial_cds_chars.len() {
                        let nc: Vec<char> = new_codon.chars().collect();
                        initial_cds_chars[start] = nc[0];
                        initial_cds_chars[start + 1] = nc[1];
                        initial_cds_chars[start + 2] = nc[2];
                    }
                }
            }
            let initial_cds: String = initial_cds_chars.into_iter().collect();
            let mut mutable_pos_0based: Vec<usize> = mutations_pos.iter().map(|p| p - 1).collect();
            if let Some(region_positions) =
                parse_mcmc_region(&opt.mcmc_region, cds_seq.len(), false)
            {
                let region_set: std::collections::HashSet<usize> =
                    region_positions.into_iter().collect();
                mutable_pos_0based.retain(|p| region_set.contains(p));
            }
            let mut searcher = MCMCSearch::new(
                idx.clone(),
                aa_seq_vec,
                initial_cds,
                codon_table.clone(),
                config,
                Arc::clone(&avoid_seqs),
                opt.mcmc_iterations,
                opt.mcmc_mutations,
                opt.mcmc_accept_prob,
                opt.mcmc_temperature,
                Some(mutable_pos_0based),
                no_verbose,
                0,
                opt.mcmc_traces,
                Arc::clone(&trace_pool),
            );
            searcher.search().into_iter().next().unwrap()
        } else {
            let snp_mut = SnpMutation::new(
                idx.clone(),
                mutations_pos,
                target_aa,
                CDSSeq::new(&cds_seq),
                codon_table.clone(),
                thread_pool,
                avoid_seqs,
                config,
            );
            snp_mut.search_best()
        };

        let best_out = best_sp.to_json();
        let mut content = json!({
            "results": vec![best_out],
        });
        if let Some(obj) = content.as_object_mut() {
            obj.insert(
                "params".to_string(),
                serde_json::to_value(opt)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
            );
        }
        let suffix = "snp_fit".to_string();
        save_json_content(
            content,
            opt.prefix.clone(),
            opt.outdir.clone(),
            Some(opt.lambda),
            &suffix,
            true,
        )?;
    }
    Ok(())
}

fn run_assay(opt: &AssayOptions) -> std::io::Result<()> {
    let fa_reader = FaReader::new(opt.fasta.clone(), SeqType::DNA);

    // Fail fast if PLL is requested but any sequence exceeds the model limit.
    if let Some(_) = opt.model_path {
        for (idx, seq) in &fa_reader.id_seqs_table {
            let codon_count = seq.len() / 3;
            if codon_count > MAX_PLL_CODON_COUNT {
                eprintln!(
                    "[codon_pll] Error: sequence '{}' has {} codons, exceeding the PLL limit of {} codons (~{} amino acids with stop codon).",
                    idx.trim_start_matches('>'), codon_count, MAX_PLL_CODON_COUNT, MAX_PLL_AA_LEN
                );
                std::process::exit(1);
            }
        }
    }

    let mut rna_reader = FaReader::new(opt.fasta.clone(), SeqType::DNA);
    for (_idx, dna) in &mut rna_reader.id_seqs_table {
        *dna = dna.replace("T", "U");
    }

    let mut palin_cals = Vec::new();
    for (idx, rna) in &rna_reader.id_seqs_table {
        palin_cals.push(PalinCalculator::new(idx, rna));
    }
    let palin_outs = palin_cals
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();

    let mut m2s_cals = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        m2s_cals.push(M2sCalculator::new(idx, seq));
    }
    let m2s_outs = m2s_cals
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();

    let mut gc_cals = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        gc_cals.push(GcCalculator::new(idx, seq));
    }
    let gc_outs = gc_cals.into_iter().map(|x| x.to_json()).collect::<Vec<_>>();

    configure_mfe_window(opt.mfe_window);
    let mut mfe_cals = Vec::new();
    for (idx, seq) in &rna_reader.id_seqs_table {
        mfe_cals.push(MfeCalculator::new(idx, seq, opt.mfe_method));
    }
    let mfe_outs = mfe_cals
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();

    let mut cai_cals = Vec::new();
    if !check_table_short(&opt.freq_table_info) {
        println!("The provided short name must be one of {:?}", LEGAL_SHORTS);
        std::process::exit(1);
    }
    if let Err(e) = load_codon_table(&opt.freq_table_info) {
        println!("Error loading custom codon table: {}", e);
        std::process::exit(1);
    }
    let codon_table = CodonTables::short2enum(&opt.freq_table_info);
    for (idx, seq) in &fa_reader.id_seqs_table {
        cai_cals.push(RawCaiCalculator::new(idx, seq, codon_table.clone()));
    }
    let cai_outs = cai_cals
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();

    let mut content = json!({
        "palindrome": palin_outs,
        "mfe": mfe_outs,
        "gc": gc_outs,
        "m2s": m2s_outs,
        "cai": cai_outs,
    });

    if let Some(ref model_path) = opt.model_path {
        let model_path_str = model_path.to_string_lossy().to_string();
        let mut pll_cals = Vec::new();
        for (idx, seq) in &fa_reader.id_seqs_table {
            pll_cals.push(CodonPllCalculator::new(idx, seq, model_path_str.clone()));
        }
        let pll_outs = pll_cals
            .into_iter()
            .map(|x| x.to_json())
            .collect::<Vec<_>>();
        content["pll"] = json!(pll_outs);
    }

    let suffix = "assay";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        &suffix,
        false,
    )?;
    Ok(())
}

fn run_pll(opt: &CodonPllOptions) -> std::io::Result<()> {
    let seq_type = SeqType::DNA;
    let fa_reader = FaReader::new(opt.fasta.clone(), seq_type);
    for (idx, seq) in &fa_reader.id_seqs_table {
        let codon_count = seq.len() / 3;
        if codon_count > MAX_PLL_CODON_COUNT {
            eprintln!(
                "[codon_pll] Error: sequence '{}' has {} codons, exceeding the PLL limit of {} codons (~{} amino acids with stop codon).",
                idx.trim_start_matches('>'), codon_count, MAX_PLL_CODON_COUNT, MAX_PLL_AA_LEN
            );
            std::process::exit(1);
        }
    }
    let model_path = opt.model_path.to_string_lossy().to_string();
    let mut calculators = Vec::new();
    for (idx, seq) in &fa_reader.id_seqs_table {
        calculators.push(CodonPllCalculator::new(idx, seq, model_path.clone()));
    }
    let pll_outs = calculators
        .into_iter()
        .map(|x| x.to_json())
        .collect::<Vec<_>>();
    let content = json!({
        "results": pll_outs,
    });
    let suffix = "codon_pll";
    save_json_content(
        content,
        opt.prefix.clone(),
        opt.outdir.clone(),
        None,
        suffix,
        false,
    )?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = SubCommand::from_args();
    match args {
        SubCommand::Fit(ref opt) => run_fit(opt),
        SubCommand::Ufit(ref opt) => run_ufit(opt),
        SubCommand::Cai(ref opt) => run_cai(opt),
        SubCommand::Mfe(ref opt) => run_mfe(opt),
        SubCommand::Gc(ref opt) => run_gc(opt),
        SubCommand::Palindrome(ref opt) => run_palindrome(opt),
        SubCommand::Mutate2Stop(ref opt) => run_mutate2stop(opt),
        SubCommand::SnpFit(ref opt) => run_snp(opt),
        SubCommand::Assay(ref opt) => run_assay(opt),
        SubCommand::CodonPll(ref opt) => run_pll(opt),
        SubCommand::Explore(ref opt) => run_explore(opt),
        SubCommand::RunConfig(ref opt) => {
            if opt.print_default_config {
                config::print_default_config();
                return Ok(());
            }
            let config_path = opt.config.clone().unwrap_or_else(|| {
                eprintln!("Error: --config <path> is required (unless --print-default-config)");
                std::process::exit(1);
            });
            let content = std::fs::read_to_string(&config_path)?;
            let config: ConfigFile = toml::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            config.run()
        }
    }
}

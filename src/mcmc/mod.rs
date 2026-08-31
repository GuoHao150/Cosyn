//! MCMC-based codon optimization module.
//!
//! Provides an alternative search strategy to beam search. Given any input
//! sequence (DNA, RNA, or amino acid), the algorithm first converts it to a
//! CDS. It then iteratively proposes random synonymous codon substitutions
//! and accepts/rejects them according to a Metropolis-like criterion. This
//! tends to explore a broader, less extreme region of the Pareto front than
//! deterministic beam search.

use crate::beam_search::{CodonTables, OptimizeObject, SearchConfig, SearchPath};
use crate::cai::CaiMode;
use crate::cds::{LossFuncs, LossInfo, Triplet};
use crate::codon_pll_binding::{cds_to_codon_ids, eval_pll};
use crate::mfe_binding::MfeMethod;
use crate::seq_patterns::seq_patterns::AvoidSeqs;
use num_cpus;
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::mpsc;
use std::sync::Arc;
use threadpool::ThreadPool;

/// Convert a DNA CDS string into a `SearchPath` so that the existing loss
/// functions in `LossFuncs` can be reused without modification.
pub fn cds_to_search_path(cds: &str, seq_id: &str, codon_table: CodonTables) -> SearchPath {
    let chars: Vec<char> = cds.chars().collect();
    let mut triplets = Vec::with_capacity(chars.len() / 3);
    for chunk in chars.chunks(3) {
        if chunk.len() == 3 {
            triplets.push(Triplet::new(chunk[0], chunk[1], chunk[2]));
        }
    }
    SearchPath::new(triplets, seq_id.to_string(), codon_table)
}

/// Evaluate a `SearchPath` using the same loss-function dispatch table as
/// beam search. Keeping the dispatch logic identical ensures that MCMC and
/// beam search share identical loss semantics.
pub fn evaluate_search_path(sp: &SearchPath, config: &SearchConfig) -> LossInfo {
    let codon_table = sp.codon_table.get_codon_table();
    let mut loss = match config.opt_obj {
        OptimizeObject::MfeCai => LossFuncs::loss_cai_mfe(
            sp,
            config.lambda,
            codon_table,
            config.mfe_method.clone(),
            false,
            config.cai_mode,
        ),
        OptimizeObject::GcCai => LossFuncs::loss_cai_gc(sp, codon_table, config.cai_mode),
        OptimizeObject::MfeGcCai => LossFuncs::loss_cai_mfe_gc(
            config.lambda,
            sp,
            codon_table,
            config.mfe_method.clone(),
            false,
            config.cai_mode,
        ),
        OptimizeObject::MfeCaiPalin => LossFuncs::loss_cai_mfe_palin(
            sp,
            config.lambda,
            codon_table,
            config.mfe_method.clone(),
            config.weight_cai,
            config.weight_palindrome,
            false,
            config.cai_mode,
        ),
        OptimizeObject::GcCaiPalin => LossFuncs::loss_cai_gc_palin(
            sp,
            codon_table,
            config.weight_cai,
            config.weight_gc,
            config.weight_palindrome,
            config.cai_mode,
        ),
        OptimizeObject::MfeGcCaiPalin => LossFuncs::loss_cai_mfe_gc_palin(
            config.lambda,
            sp,
            codon_table,
            config.mfe_method.clone(),
            config.weight_cai,
            config.weight_gc,
            config.weight_palindrome,
            false,
            config.cai_mode,
        ),
        OptimizeObject::MfeCaiGcM2sPalin => LossFuncs::loss_cai_mfe_gc_m2s_palin(
            config.lambda,
            sp,
            codon_table,
            config.mfe_method.clone(),
            config.weight_cai,
            config.weight_gc,
            config.weight_m2s,
            config.weight_palindrome,
            false,
            config.cai_mode,
        ),
        OptimizeObject::CaiGcM2sPalin => LossFuncs::loss_cai_gc_m2s_palin(
            config.lambda,
            sp,
            codon_table,
            config.weight_cai,
            config.weight_m2s,
            config.weight_gc,
            config.weight_palindrome,
            config.cai_mode,
        ),
        OptimizeObject::UnifiedObj => LossFuncs::loss_unified(
            config.lambda,
            sp,
            codon_table,
            config.mfe_method.clone(),
            1, // default weight_mfe for UnifiedObj when not provided via UfitOptions
            config.weight_cai,
            config.weight_gc,
            config.weight_m2s,
            config.weight_palindrome,
            config.model_path.as_deref(),
            config.weight_pll,
            false,
            config.cai_mode,
        ),
    };
    let rna = sp.get_rna();
    loss
}

/// Re-evaluate a candidate sequence with the full set of reporting metrics.
///
/// During MCMC the search objective may omit some terms (e.g. `--pmgc` ignores
/// MFE). However, the final JSON/FASTA output should always report the five
/// basic metrics: CAI, GC%, MFE, palindrome score, and mutate-to-stop count.
/// If a CodonTransformer model path is provided, the codon PLL is also computed
/// and reported.
///
/// The `loss` field is preserved from the search objective so that the reported
/// total loss reflects what was actually optimized.
pub fn fill_complete_metrics(
    sp: &SearchPath,
    search_loss: &LossInfo,
    config: &SearchConfig,
) -> LossInfo {
    let codon_table = sp.codon_table.get_codon_table();

    // Compute the five basic metrics, respecting --mfe_pll_start if the user
    // requested to skip MFE/PLL during design.
    let skip_mfe_pll = sp.triplet_seq.len() < config.mfe_pll_start_codon;
    let full = LossFuncs::loss_cai_mfe_gc_m2s_palin(
        config.lambda,
        sp,
        codon_table,
        config.mfe_method.clone(),
        config.weight_cai,
        config.weight_gc,
        config.weight_m2s,
        config.weight_palindrome,
        skip_mfe_pll,
        CaiMode::Scaled,
    );

    // Preserve the loss that was actually used during search, but fill in
    // all reporting metrics. Fields are public within the crate.
    let (pll_score, raw_pll) = if let Some(model_path) = config.model_path.as_ref() {
        let seq = sp.get_cds();
        let ids = cds_to_codon_ids(&seq);
        let raw = eval_pll(model_path, &ids);
        // pll_score = negated * weight (for unified loss tracking)
        // raw_pll  = original model output (for user-facing display)
        (raw * -1.0 * config.weight_pll, raw)
    } else {
        (0.0, 0.0)
    };

    LossInfo {
        loss: search_loss.loss,
        mfe: full.mfe,
        scaled_cai: full.scaled_cai,
        second: full.second,
        palindrome_score: full.palindrome_score,
        palindrome_seqs: full.palindrome_seqs,
        m2s_score: full.m2s_score,
        m2s_nums: full.m2s_nums,
        pll_score,
        raw_pll,
    }
}

/// Sample a synonymous codon for an amino acid according to the codon table
/// frequencies. Frequencies are treated as unnormalized weights.
pub(crate) fn sample_synonymous_codon<R: Rng>(
    aa: &str,
    codon_table: &CodonTables,
    rng: &mut R,
) -> Option<String> {
    use crate::cai::{aa_to_idx, CODONS_FOR_AA, CODON_STRINGS};

    let aa_idx = aa_to_idx(aa);
    let codon_indices = CODONS_FOR_AA[aa_idx as usize];
    if codon_indices.is_empty() {
        return None;
    }

    let table = codon_table.get_codon_table();
    let pairs: Vec<(u8, f64)> = codon_indices
        .iter()
        .map(|&idx| (idx, table.freq_by_idx(idx)))
        .collect();
    let total: f64 = pairs.iter().map(|(_, f)| f).sum();
    if total <= 0.0 {
        // Fallback to uniform if frequencies are missing/invalid
        let idx = *codon_indices.choose(rng)?;
        return Some(CODON_STRINGS[idx as usize].to_string());
    }
    let mut pick = rng.gen::<f64>() * total;
    for (idx, freq) in &pairs {
        pick -= *freq;
        if pick <= 0.0 {
            return Some(CODON_STRINGS[*idx as usize].to_string());
        }
    }
    let idx = pairs.last()?.0;
    Some(CODON_STRINGS[idx as usize].to_string())
}

/// Lightweight runner for a single MCMC trace.
///
/// Cloned from `MCMCSearch` before being dispatched to the thread pool.
/// Contains only the fields needed for proposal + evaluation, omitting
/// UI state (`tqdm` bar position, `no_verbose`) and the thread pool itself.
#[derive(Clone)]
struct SingleTraceRunner {
    seq_id: String,
    aa_seq: Vec<String>,
    codon_table: CodonTables,
    config: SearchConfig,
    avoid_seqs: Arc<AvoidSeqs>,
    iterations: usize,
    mutation_count: usize,
    accept_prob: f64,
    temperature: f64,
    mutable_positions: Option<Vec<usize>>,
}

impl SingleTraceRunner {
    fn propose_from<R: Rng>(&self, base_cds: &str, rng: &mut R) -> String {
        let num_codons = self.aa_seq.len();
        if num_codons == 0 {
            return base_cds.to_string();
        }
        let candidate_positions: Vec<usize> = match &self.mutable_positions {
            Some(pos) if !pos.is_empty() => pos.clone(),
            _ => (0..num_codons).collect(),
        };
        let n_mut = self.mutation_count.min(candidate_positions.len()).max(1);
        let mut selected: Vec<usize> = candidate_positions
            .choose_multiple(rng, n_mut)
            .copied()
            .collect();
        selected.sort_unstable();

        let mut new_cds = base_cds.to_string();
        for &pos in &selected {
            let start = pos * 3;
            if start + 2 >= new_cds.len() {
                continue;
            }
            let old_codon = &new_cds[start..start + 3];
            let aa = &self.aa_seq[pos];
            if let Some(new_codon) = sample_synonymous_codon(aa, &self.codon_table, rng) {
                if new_codon != old_codon {
                    new_cds.replace_range(start..start + 3, &new_codon);
                }
            }
        }
        new_cds
    }

    fn is_valid(&self, cds: &str) -> bool {
        let s = cds.to_string();
        self.avoid_seqs.filter_cds(&s) && self.avoid_seqs.filter_homopolymers(&s)
    }

    fn accept_move<R: Rng>(&self, old_loss: f64, new_loss: f64, rng: &mut R) -> bool {
        let improved = if self.config.if_minimize_loss {
            new_loss <= old_loss
        } else {
            new_loss >= old_loss
        };
        if improved {
            return true;
        }

        let delta = if self.config.if_minimize_loss {
            new_loss - old_loss
        } else {
            old_loss - new_loss
        };

        if self.temperature > 0.0 {
            let p = (-delta / self.temperature).exp();
            rng.gen::<f64>() < p
        } else {
            rng.gen::<f64>() < self.accept_prob
        }
    }

    fn run(&self, initial_cds: String, initial_loss: LossInfo) -> SearchPath {
        let mut rng = rand::thread_rng();
        let mut current_cds = initial_cds.clone();
        let mut current_loss_val = initial_loss.loss;
        let mut best_cds = initial_cds;
        let mut best_loss = initial_loss;

        for _ in 0..self.iterations {
            let proposed_cds = self.propose_from(&current_cds, &mut rng);

            if !self.is_valid(&proposed_cds) {
                continue;
            }

            let proposed_sp =
                cds_to_search_path(&proposed_cds, &self.seq_id, self.codon_table.clone());
            let proposed_loss = evaluate_search_path(&proposed_sp, &self.config);
            let proposed_loss_val = proposed_loss.loss;

            if self.accept_move(current_loss_val, proposed_loss_val, &mut rng) {
                current_cds = proposed_cds;
                current_loss_val = proposed_loss_val;

                let is_better = if self.config.if_minimize_loss {
                    proposed_loss_val < best_loss.loss
                } else {
                    proposed_loss_val > best_loss.loss
                };
                if is_better {
                    best_cds = current_cds.clone();
                    best_loss = proposed_loss;
                }
            }
        }

        // Final re-evaluation to fill all reporting metrics
        let best_sp = cds_to_search_path(&best_cds, &self.seq_id, self.codon_table.clone());
        let search_loss = evaluate_search_path(&best_sp, &self.config);
        let complete_loss = fill_complete_metrics(&best_sp, &search_loss, &self.config);
        SearchPath::update_loss_value(best_sp, complete_loss)
    }
}

/// MCMC search state and driver.
#[derive(Clone)]
pub struct MCMCSearch {
    pub seq_id: String,
    /// Amino-acid sequence (one element per residue). Used during mutation
    /// to ensure codon substitutions are synonymous.
    pub aa_seq: Vec<String>,
    pub current_cds: String,
    pub current_loss: Option<LossInfo>,
    pub best_cds: String,
    pub best_loss: Option<LossInfo>,
    pub codon_table: CodonTables,
    pub config: SearchConfig,
    pub avoid_seqs: Arc<AvoidSeqs>,
    pub iterations: usize,
    /// Number of codons randomly mutated in each MCMC proposal.
    pub mutation_count: usize,
    /// Fixed acceptance probability for worse proposals (used when
    /// `temperature == 0.0`).
    pub accept_prob: f64,
    /// If positive, use Boltzmann acceptance `exp(-ΔE / T)` instead of the
    /// fixed-probability rule.
    pub temperature: f64,
    /// Optional restriction of mutation positions (0-based codon indices).
    /// If `None`, all codon positions are mutable. This is useful for SNP
    /// optimization where only specified residue positions should be changed.
    pub mutable_positions: Option<Vec<usize>>,
    /// Suppress progress bar output
    pub no_verbose: bool,
    /// Progress bar position (for multi-sequence parallel display)
    pub bar_position: u16,
    /// Number of independent MCMC chains to run
    pub traces: usize,
    /// Thread pool for parallel trace execution
    pub thread_pool: Arc<ThreadPool>,
}

impl MCMCSearch {
    pub fn new(
        seq_id: String,
        aa_seq: Vec<String>,
        initial_cds: String,
        codon_table: CodonTables,
        config: SearchConfig,
        avoid_seqs: Arc<AvoidSeqs>,
        iterations: usize,
        mutation_count: usize,
        accept_prob: f64,
        temperature: f64,
        mutable_positions: Option<Vec<usize>>,
        no_verbose: bool,
        bar_position: u16,
        traces: usize,
        thread_pool: Arc<ThreadPool>,
    ) -> Self {
        let sp = cds_to_search_path(&initial_cds, &seq_id, codon_table.clone());
        let loss = evaluate_search_path(&sp, &config);
        MCMCSearch {
            seq_id,
            aa_seq,
            current_cds: initial_cds.clone(),
            current_loss: Some(loss.clone()),
            best_cds: initial_cds,
            best_loss: Some(loss),
            codon_table,
            config,
            avoid_seqs,
            iterations,
            mutation_count,
            accept_prob,
            temperature,
            mutable_positions,
            no_verbose,
            bar_position,
            traces,
            thread_pool,
        }
    }

    /// Build a `SingleTraceRunner` from this searcher's current configuration.
    fn to_runner(&self) -> SingleTraceRunner {
        SingleTraceRunner {
            seq_id: self.seq_id.clone(),
            aa_seq: self.aa_seq.clone(),
            codon_table: self.codon_table.clone(),
            config: self.config.clone(),
            avoid_seqs: Arc::clone(&self.avoid_seqs),
            iterations: self.iterations,
            mutation_count: self.mutation_count,
            accept_prob: self.accept_prob,
            temperature: self.temperature,
            mutable_positions: self.mutable_positions.clone(),
        }
    }

    /// Run `self.traces` independent MCMC chains in parallel via the thread pool.
    /// Returns a `Vec<SearchPath>` containing the best candidate from each
    /// chain, sorted by loss (best first).
    pub fn search(&mut self) -> Vec<SearchPath> {
        let initial_cds = self.current_cds.clone();
        let initial_loss = self.current_loss.as_ref().cloned().unwrap_or_else(|| {
            let sp = cds_to_search_path(&initial_cds, &self.seq_id, self.codon_table.clone());
            evaluate_search_path(&sp, &self.config)
        });

        let n_traces = self.traces;
        let (tx, rx) = mpsc::channel();

        if !self.no_verbose {
            eprintln!(
                "MCMC {}: dispatching {} traces ({} iter each) to thread pool...",
                &self.seq_id, n_traces, self.iterations
            );
        }

        for _trace_idx in 0..n_traces {
            let tx = tx.clone();
            let runner = self.to_runner();
            let cds = initial_cds.clone();
            let loss = initial_loss.clone();

            self.thread_pool.execute(move || {
                let result = runner.run(cds, loss);
                tx.send(result).ok();
            });
        }
        drop(tx); // close the channel after all senders are dropped

        let mut all_results: Vec<SearchPath> = Vec::with_capacity(n_traces);
        for _i in 0..n_traces {
            match rx.recv() {
                Ok(result) => {
                    all_results.push(result);
                    if !self.no_verbose {
                        eprintln!(
                            "MCMC {}: trace {}/{} completed",
                            &self.seq_id,
                            all_results.len(),
                            n_traces
                        );
                    }
                }
                Err(_) => break,
            }
        }

        // Sort results: best loss first
        if self.config.if_minimize_loss {
            all_results.sort_by(|a, b| {
                a.loss_value
                    .as_ref()
                    .unwrap()
                    .loss
                    .partial_cmp(&b.loss_value.as_ref().unwrap().loss)
                    .unwrap()
            });
        } else {
            all_results.sort_by(|a, b| {
                b.loss_value
                    .as_ref()
                    .unwrap()
                    .loss
                    .partial_cmp(&a.loss_value.as_ref().unwrap().loss)
                    .unwrap()
            });
        }

        // Update self fields with the overall best result
        if let Some(best) = all_results.first() {
            self.best_cds = best.get_cds();
            self.best_loss = best.loss_value.clone();
        }

        all_results
    }

    /// Set the progress bar position for parallel display.
    pub fn set_bar_position(mut self, new_pos: u16) -> Self {
        self.bar_position = new_pos;
        self
    }
}

/// Two-phase MCMC optimization for `--weak-head`.
///
/// **Phase 1** — Design the 5'-end head region (first `weak_head_bases` nt)
/// to avoid stable secondary structure by maximizing MFE (making it as close
/// to zero as possible). Only head codons are mutable.
///
/// **Phase 2** — Design the remaining tail region using the user's original
/// optimization parameters. All tail codons are mutable.
///
/// **Phase 3** — Concatenate head + tail and compute full reporting metrics
/// (CAI, GC%, MFE, palindrome, mutate-to-stop, and optional PLL).
pub fn run_weak_head_mcmc(
    seq_id: &str,
    aa_seq: &[String],
    input_cds: &str,
    codon_table: &CodonTables,
    tail_config: &SearchConfig,
    avoid_seqs: Arc<AvoidSeqs>,
    iterations: usize,
    mutation_count: usize,
    accept_prob: f64,
    temperature: f64,
    traces: usize,
    weak_head_bases: usize,
    mfe_method: &MfeMethod,
    bar_position: u16,
    no_verbose: bool,
) -> Vec<SearchPath> {
    let head_bases = weak_head_bases;
    let head_codons = head_bases / 3;
    let total_codons = aa_seq.len();

    // If head covers the entire (or longer) sequence, run a single MCMC with
    // the weak-head penalty embedded in the loss function via SearchConfig.
    if head_codons >= total_codons {
        let mut full_config = tail_config.clone();
        full_config.weak_head_bases = head_bases;
        let sp = cds_to_search_path(input_cds, seq_id, codon_table.clone());
        let loss = evaluate_search_path(&sp, &full_config);
        let complete = fill_complete_metrics(&sp, &loss, &full_config);
        return vec![SearchPath::update_loss_value(sp, complete)];
    }

    // ── Phase 1: Design head to avoid stable secondary structure ──────────
    let head_aa = aa_seq[..head_codons].to_vec();
    let head_input_cds = &input_cds[..head_bases];

    let head_config = SearchConfig {
        lambda: 0.0, // MFE-only objective
        mfe_method: *mfe_method,
        opt_obj: OptimizeObject::MfeCai,
        if_minimize_loss: false, // maximize → find least-stable (highest MFE)
        weak_head_bases: 0,      // no recursive weak-head
        ..tail_config.clone()
    };

    let mut head_searcher = MCMCSearch::new(
        format!("{}_head", seq_id),
        head_aa,
        head_input_cds.to_string(),
        codon_table.clone(),
        head_config,
        Arc::clone(&avoid_seqs),
        iterations.max(200),
        mutation_count.min(head_codons).max(1),
        accept_prob,
        temperature,
        Some((0..head_codons).collect()),
        no_verbose,
        bar_position,
        traces.max(1),
        Arc::new(ThreadPool::new(num_cpus::get())),
    );
    let head_results = head_searcher.search();
    let best_head_cds = head_results[0].get_cds();

    // ── Phase 2: Design tail with user's original parameters ──────────────
    let tail_codons = total_codons - head_codons;
    let tail_aa = aa_seq[head_codons..].to_vec();
    let tail_input_cds = &input_cds[head_bases..];

    let mut tail_config_clean = tail_config.clone();
    tail_config_clean.weak_head_bases = 0;

    let mut tail_searcher = MCMCSearch::new(
        format!("{}_tail", seq_id),
        tail_aa,
        tail_input_cds.to_string(),
        codon_table.clone(),
        tail_config_clean,
        Arc::clone(&avoid_seqs),
        iterations,
        mutation_count.min(tail_codons).max(1),
        accept_prob,
        temperature,
        None, // all tail positions mutable
        no_verbose,
        bar_position.wrapping_add(1),
        traces,
        Arc::new(ThreadPool::new(num_cpus::get())),
    );
    let tail_results = tail_searcher.search();
    let best_tail_cds = tail_results[0].get_cds();

    // ── Phase 3: Concatenate and evaluate full metrics ────────────────────
    let full_cds = format!("{}{}", best_head_cds, best_tail_cds);
    let full_sp = cds_to_search_path(&full_cds, seq_id, codon_table.clone());

    // Use the user's original config for reporting metrics
    let search_loss = evaluate_search_path(&full_sp, tail_config);
    let complete_loss = fill_complete_metrics(&full_sp, &search_loss, tail_config);

    vec![SearchPath::update_loss_value(full_sp, complete_loss)]
}

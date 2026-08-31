use crate::cai::CaiMode;
use crate::cli::*;
use crate::mfe_binding::MfeMethod;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level TOML configuration for `cosyn run-config`.
///
/// The `command` field selects which subcommand to run. Only the matching
/// section is required; all other sections are ignored.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ConfigFile {
    pub command: String,

    #[serde(default)]
    pub fit: Option<FitConfig>,
    #[serde(default)]
    pub ufit: Option<UfitConfig>,
    #[serde(default)]
    pub cai: Option<CaiConfig>,
    #[serde(default)]
    pub mfe: Option<MfeConfig>,
    #[serde(default)]
    pub gc: Option<GcConfig>,
    #[serde(default)]
    pub palindrome: Option<PalindromeConfig>,
    #[serde(default)]
    pub mutate2stop: Option<M2sConfig>,
    #[serde(default)]
    pub snp: Option<SnpConfig>,
    #[serde(default)]
    pub assay: Option<AssayConfig>,
    #[serde(default)]
    pub pll: Option<CodonPllConfig>,
    #[serde(default)]
    pub explore: Option<ExploreConfig>,
}

impl ConfigFile {
    /// Dispatch to the requested subcommand implementation.
    pub fn run(self) -> std::io::Result<()> {
        match self.command.as_str() {
            "fit" => crate::run_fit(&self.fit.unwrap_or_default().into_options()),
            "ufit" => crate::run_ufit(&self.ufit.unwrap_or_default().into_options()),
            "cai" => crate::run_cai(&self.cai.unwrap_or_default().into_options()),
            "mfe" => crate::run_mfe(&self.mfe.unwrap_or_default().into_options()),
            "gc" => crate::run_gc(&self.gc.unwrap_or_default().into_options()),
            "palindrome" => {
                crate::run_palindrome(&self.palindrome.unwrap_or_default().into_options())
            }
            "mutate2stop" => {
                crate::run_mutate2stop(&self.mutate2stop.unwrap_or_default().into_options())
            }
            "snp" => crate::run_snp(&self.snp.unwrap_or_default().into_options()),
            "assay" => crate::run_assay(&self.assay.unwrap_or_default().into_options()),
            "pll" => crate::run_pll(&self.pll.unwrap_or_default().into_options()),
            "explore" => crate::run_explore(&self.explore.unwrap_or_default().into_options()),
            other => {
                eprintln!(
                    "Error: unknown command '{}' in configuration file. Supported: fit, ufit, cai, mfe, gc, palindrome, mutate2stop, snp, assay, pll, explore",
                    other
                );
                std::process::exit(1);
            }
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FitConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub win_size: Option<u32>,
    #[serde(default)]
    pub rnd_size: Option<u32>,
    #[serde(default)]
    pub num_outputs: Option<u32>,
    #[serde(default)]
    pub lambda: Option<f64>,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub seq_parallel: Option<u32>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub is_aa_seq: Option<bool>,
    #[serde(default)]
    pub is_rna: Option<bool>,
    #[serde(default)]
    pub homopolymers: Option<usize>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub avoid_seqs: Option<PathBuf>,
    #[serde(default)]
    pub freq_table_info: Option<String>,
    #[serde(default)]
    pub gc_cai: Option<bool>,
    #[serde(default)]
    pub mfe_cai: Option<bool>,
    #[serde(default)]
    pub no_verbose: Option<bool>,
    #[serde(default)]
    pub palindrome: Option<bool>,
    #[serde(default)]
    pub cmgmp: Option<bool>,
    #[serde(default)]
    pub pmgc: Option<bool>,
    #[serde(default)]
    pub weight_gc: Option<f64>,
    #[serde(default)]
    pub weight_cai: Option<f64>,
    #[serde(default)]
    pub weight_m2s: Option<f64>,
    #[serde(default)]
    pub weight_palindrome: Option<f64>,
    #[serde(default)]
    pub maximize_loss: Option<bool>,
    #[serde(default)]
    pub mfe_pll_start_codon: Option<usize>,
    #[serde(default)]
    pub search_method: Option<String>,
    #[serde(default)]
    pub mcmc_iterations: Option<usize>,
    #[serde(default)]
    pub mcmc_mutations: Option<usize>,
    #[serde(default)]
    pub mcmc_accept_prob: Option<f64>,
    #[serde(default)]
    pub mcmc_temperature: Option<f64>,
    #[serde(default)]
    pub mcmc_traces: Option<usize>,
    #[serde(default)]
    pub mcmc_region: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
    #[serde(default)]
    pub weak_head: Option<usize>,
    #[serde(default)]
    pub cai_mode: Option<CaiMode>,
}

impl FitConfig {
    pub fn into_options(self) -> FitOptions {
        FitOptions {
            fasta: require_path(self.fasta, "fit.fasta"),
            win_size: self.win_size.unwrap_or(50),
            rnd_size: self.rnd_size.unwrap_or(2),
            num_outputs: self.num_outputs.unwrap_or(1),
            lambda: self.lambda.unwrap_or(3.0),
            threads: optional_threads(self.threads),
            seq_parallel: self.seq_parallel.unwrap_or(1),
            outdir: optional_outdir(self.outdir),
            is_aa_seq: self.is_aa_seq.unwrap_or(false),
            is_rna: self.is_rna.unwrap_or(false),
            homopolymers: self.homopolymers.unwrap_or(6),
            prefix: optional_string(self.prefix),
            avoid_seqs: optional_path(self.avoid_seqs),
            freq_table_info: optional_nonempty_string(self.freq_table_info, "hc"),
            gc_cai: self.gc_cai.unwrap_or(false),
            mfe_cai: self.mfe_cai.unwrap_or(false),
            no_verbose: self.no_verbose.unwrap_or(false),
            palindrome: self.palindrome.unwrap_or(false),
            cmgmp: self.cmgmp.unwrap_or(false),
            pmgc: self.pmgc.unwrap_or(false),
            weight_gc: self.weight_gc.unwrap_or(1.0),
            weight_cai: self.weight_cai.unwrap_or(1.0),
            weight_m2s: self.weight_m2s.unwrap_or(1.0),
            weight_palindrome: self.weight_palindrome.unwrap_or(1.0),
            maximize_loss: self.maximize_loss.unwrap_or(false),
            mfe_pll_start_codon: self.mfe_pll_start_codon.unwrap_or(0),
            search_method: optional_nonempty_string(self.search_method, "beam"),
            mcmc_iterations: self.mcmc_iterations.unwrap_or(1000),
            mcmc_mutations: self.mcmc_mutations.unwrap_or(5),
            mcmc_accept_prob: self.mcmc_accept_prob.unwrap_or(0.5),
            mcmc_temperature: self.mcmc_temperature.unwrap_or(0.0),
            mcmc_traces: self.mcmc_traces.unwrap_or(1),
            mcmc_region: optional_string(self.mcmc_region),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(100),
            weak_head: self.weak_head.unwrap_or(0),
            cai_mode: self.cai_mode.unwrap_or(CaiMode::Scaled),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct UfitConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub win_size: Option<u32>,
    #[serde(default)]
    pub rnd_size: Option<u32>,
    #[serde(default)]
    pub num_outputs: Option<u32>,
    #[serde(default)]
    pub lambda: Option<f64>,
    #[serde(default)]
    pub weight_gc: Option<f64>,
    #[serde(default)]
    pub weight_cai: Option<f64>,
    #[serde(default)]
    pub weight_m2s: Option<f64>,
    #[serde(default)]
    pub weight_palindrome: Option<f64>,
    #[serde(default)]
    pub weight_mfe: Option<i64>,
    #[serde(default)]
    pub weight_pll: Option<f64>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub seq_parallel: Option<u32>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub is_aa_seq: Option<bool>,
    #[serde(default)]
    pub is_rna: Option<bool>,
    #[serde(default)]
    pub homopolymers: Option<usize>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub avoid_seqs: Option<PathBuf>,
    #[serde(default)]
    pub freq_table_info: Option<String>,
    #[serde(default)]
    pub maximize_loss: Option<bool>,
    #[serde(default)]
    pub no_verbose: Option<bool>,
    #[serde(default)]
    pub mfe_pll_start_codon: Option<usize>,
    #[serde(default)]
    pub search_method: Option<String>,
    #[serde(default)]
    pub mcmc_iterations: Option<usize>,
    #[serde(default)]
    pub mcmc_mutations: Option<usize>,
    #[serde(default)]
    pub mcmc_accept_prob: Option<f64>,
    #[serde(default)]
    pub mcmc_temperature: Option<f64>,
    #[serde(default)]
    pub mcmc_traces: Option<usize>,
    #[serde(default)]
    pub mcmc_region: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
    #[serde(default)]
    pub weak_head: Option<usize>,
    #[serde(default)]
    pub pll_batch_size: Option<usize>,
    #[serde(default)]
    pub pll_timeout_ms: Option<usize>,
    #[serde(default)]
    pub pll_threads: Option<usize>,
    #[serde(default)]
    pub cai_mode: Option<CaiMode>,
}

impl UfitConfig {
    pub fn into_options(self) -> UfitOptions {
        UfitOptions {
            fasta: require_path(self.fasta, "ufit.fasta"),
            win_size: self.win_size.unwrap_or(50),
            rnd_size: self.rnd_size.unwrap_or(2),
            num_outputs: self.num_outputs.unwrap_or(1),
            lambda: self.lambda.unwrap_or(3.0),
            weight_gc: self.weight_gc.unwrap_or(1.0),
            weight_cai: self.weight_cai.unwrap_or(1.0),
            weight_m2s: self.weight_m2s.unwrap_or(1.0),
            weight_palindrome: self.weight_palindrome.unwrap_or(1.0),
            weight_mfe: self.weight_mfe.unwrap_or(1),
            weight_pll: self.weight_pll.unwrap_or(0.0),
            model_path: optional_path(self.model_path),
            threads: optional_threads(self.threads),
            seq_parallel: self.seq_parallel.unwrap_or(1),
            outdir: optional_outdir(self.outdir),
            is_aa_seq: self.is_aa_seq.unwrap_or(false),
            is_rna: self.is_rna.unwrap_or(false),
            homopolymers: self.homopolymers.unwrap_or(6),
            prefix: optional_string(self.prefix),
            avoid_seqs: optional_path(self.avoid_seqs),
            freq_table_info: optional_nonempty_string(self.freq_table_info, "hc"),
            maximize_loss: self.maximize_loss.unwrap_or(false),
            no_verbose: self.no_verbose.unwrap_or(false),
            mfe_pll_start_codon: self.mfe_pll_start_codon.unwrap_or(0),
            search_method: optional_nonempty_string(self.search_method, "beam"),
            mcmc_iterations: self.mcmc_iterations.unwrap_or(1000),
            mcmc_mutations: self.mcmc_mutations.unwrap_or(5),
            mcmc_accept_prob: self.mcmc_accept_prob.unwrap_or(0.5),
            mcmc_temperature: self.mcmc_temperature.unwrap_or(0.0),
            mcmc_traces: self.mcmc_traces.unwrap_or(1),
            mcmc_region: optional_string(self.mcmc_region),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(100),
            weak_head: self.weak_head.unwrap_or(0),
            pll_batch_size: self.pll_batch_size.unwrap_or(256),
            pll_timeout_ms: self.pll_timeout_ms.unwrap_or(50),
            pll_threads: self.pll_threads.unwrap_or(0),
            cai_mode: self.cai_mode.unwrap_or(CaiMode::Arithmetic),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CaiConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub freq_table_info: Option<String>,
}

impl CaiConfig {
    pub fn into_options(self) -> CaiOptions {
        CaiOptions {
            fasta: require_path(self.fasta, "cai.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
            freq_table_info: optional_nonempty_string(self.freq_table_info, "hc"),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MfeConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
    #[serde(default)]
    pub threads: Option<u32>,
}

impl MfeConfig {
    pub fn into_options(self) -> MfeOptions {
        MfeOptions {
            fasta: require_path(self.fasta, "mfe.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(0),
            threads: self.threads,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GcConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
}

impl GcConfig {
    pub fn into_options(self) -> GcOptions {
        GcOptions {
            fasta: require_path(self.fasta, "gc.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PalindromeConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
}

impl PalindromeConfig {
    pub fn into_options(self) -> PalindromeOptions {
        PalindromeOptions {
            fasta: require_path(self.fasta, "palindrome.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct M2sConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
}

impl M2sConfig {
    pub fn into_options(self) -> M2sOptions {
        M2sOptions {
            fasta: require_path(self.fasta, "mutate2stop.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SnpConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub mutations: Option<Vec<String>>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub lambda: Option<f64>,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub homopolymers: Option<usize>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub avoid_seqs: Option<PathBuf>,
    #[serde(default)]
    pub freq_table_info: Option<String>,
    #[serde(default)]
    pub gc_cai: Option<bool>,
    #[serde(default)]
    pub mfe_cai: Option<bool>,
    #[serde(default)]
    pub palindrome: Option<bool>,
    #[serde(default)]
    pub cmgmp: Option<bool>,
    #[serde(default)]
    pub pmgc: Option<bool>,
    #[serde(default)]
    pub weight_gc: Option<f64>,
    #[serde(default)]
    pub weight_cai: Option<f64>,
    #[serde(default)]
    pub weight_palindrome: Option<f64>,
    #[serde(default)]
    pub weight_m2s: Option<f64>,
    #[serde(default)]
    pub maximize_loss: Option<bool>,
    #[serde(default)]
    pub no_verbose: Option<bool>,
    #[serde(default)]
    pub mfe_pll_start_codon: Option<usize>,
    #[serde(default)]
    pub search_method: Option<String>,
    #[serde(default)]
    pub mcmc_iterations: Option<usize>,
    #[serde(default)]
    pub mcmc_mutations: Option<usize>,
    #[serde(default)]
    pub mcmc_accept_prob: Option<f64>,
    #[serde(default)]
    pub mcmc_temperature: Option<f64>,
    #[serde(default)]
    pub mcmc_traces: Option<usize>,
    #[serde(default)]
    pub mcmc_region: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
}

impl SnpConfig {
    pub fn into_options(self) -> SnpOptions {
        let mutations = self.mutations.filter(|v| !v.is_empty()).unwrap_or_else(|| {
            eprintln!("Error: snp.mutations is required");
            std::process::exit(1);
        });
        SnpOptions {
            fasta: require_path(self.fasta, "snp.fasta"),
            mutations,
            outdir: optional_outdir(self.outdir),
            lambda: self.lambda.unwrap_or(2.0),
            threads: optional_threads(self.threads),
            homopolymers: self.homopolymers.unwrap_or(6),
            prefix: optional_string(self.prefix),
            avoid_seqs: optional_path(self.avoid_seqs),
            freq_table_info: optional_nonempty_string(self.freq_table_info, "hc"),
            gc_cai: self.gc_cai.unwrap_or(false),
            mfe_cai: self.mfe_cai.unwrap_or(false),
            palindrome: self.palindrome.unwrap_or(false),
            cmgmp: self.cmgmp.unwrap_or(false),
            pmgc: self.pmgc.unwrap_or(false),
            weight_gc: self.weight_gc.unwrap_or(1.0),
            weight_cai: self.weight_cai.unwrap_or(1.0),
            weight_palindrome: self.weight_palindrome.unwrap_or(1.0),
            weight_m2s: self.weight_m2s.unwrap_or(1.0),
            maximize_loss: self.maximize_loss.unwrap_or(false),
            no_verbose: self.no_verbose.unwrap_or(false),
            mfe_pll_start_codon: self.mfe_pll_start_codon.unwrap_or(0),
            search_method: optional_nonempty_string(self.search_method, "beam"),
            mcmc_iterations: self.mcmc_iterations.unwrap_or(1000),
            mcmc_mutations: self.mcmc_mutations.unwrap_or(5),
            mcmc_accept_prob: self.mcmc_accept_prob.unwrap_or(0.5),
            mcmc_temperature: self.mcmc_temperature.unwrap_or(0.0),
            mcmc_traces: self.mcmc_traces.unwrap_or(1),
            mcmc_region: optional_string(self.mcmc_region),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(100),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AssayConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub freq_table_info: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
}

impl AssayConfig {
    pub fn into_options(self) -> AssayOptions {
        AssayOptions {
            fasta: require_path(self.fasta, "assay.fasta"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
            freq_table_info: optional_nonempty_string(self.freq_table_info, "hc"),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(0),
            model_path: self.model_path,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CodonPllConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
}

impl CodonPllConfig {
    pub fn into_options(self) -> CodonPllOptions {
        CodonPllOptions {
            fasta: require_path(self.fasta, "pll.fasta"),
            model_path: require_path(self.model_path, "pll.model_path"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExploreConfig {
    #[serde(default)]
    pub fasta: Option<PathBuf>,
    #[serde(default)]
    pub codon_table: Option<String>,
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub mfe_method: Option<MfeMethod>,
    #[serde(default)]
    pub mfe_window: Option<usize>,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub extra_tables: Option<String>,
    #[serde(default)]
    pub no_opt: Option<bool>,
    #[serde(default)]
    pub no_intermediate: Option<bool>,
    #[serde(default)]
    pub no_deopt: Option<bool>,
    #[serde(default)]
    pub skip_mfe: Option<bool>,
    #[serde(default)]
    pub pll_weight: Option<f64>,
    #[serde(default)]
    pub pll_mcmc_iters: Option<usize>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub intermediate_iters: Option<String>,
    #[serde(default)]
    pub win_size: Option<u32>,
    #[serde(default)]
    pub lambda: Option<f64>,
    #[serde(default)]
    pub homopolymers: Option<usize>,
    #[serde(default)]
    pub avoid_seqs: Option<PathBuf>,
    #[serde(default)]
    pub is_aa: Option<bool>,
    #[serde(default)]
    pub is_rna: Option<bool>,
    #[serde(default)]
    pub no_verbose: Option<bool>,
}

impl ExploreConfig {
    pub fn into_options(self) -> ExploreOptions {
        ExploreOptions {
            fasta: require_path(self.fasta, "explore.fasta"),
            codon_table: optional_nonempty_string(self.codon_table, "hc"),
            outdir: optional_outdir(self.outdir),
            prefix: optional_string(self.prefix),
            mfe_method: self.mfe_method.unwrap_or_default(),
            mfe_window: self.mfe_window.unwrap_or(100),
            threads: optional_threads(self.threads),
            extra_tables: optional_string(self.extra_tables),
            no_opt: self.no_opt.unwrap_or(false),
            no_intermediate: self.no_intermediate.unwrap_or(false),
            no_deopt: self.no_deopt.unwrap_or(false),
            skip_mfe: self.skip_mfe.unwrap_or(false),
            pll_weight: self.pll_weight.unwrap_or(0.0),
            pll_mcmc_iters: self.pll_mcmc_iters.unwrap_or(100),
            model_path: optional_path(self.model_path),
            intermediate_iters: optional_nonempty_string(self.intermediate_iters, "100,300,1000"),
            win_size: self.win_size.unwrap_or(50),
            lambda: self.lambda.unwrap_or(3.0),
            homopolymers: self.homopolymers.unwrap_or(6),
            avoid_seqs: optional_path(self.avoid_seqs),
            is_aa: self.is_aa.unwrap_or(false),
            is_rna: self.is_rna.unwrap_or(false),
            no_verbose: self.no_verbose.unwrap_or(false),
        }
    }
}

fn require_path(path: Option<PathBuf>, name: &str) -> PathBuf {
    path.filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| {
            eprintln!("Error: config field '{}' is required", name);
            std::process::exit(1);
        })
}

fn optional_path(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|p| !p.as_os_str().is_empty())
}

fn optional_string(s: Option<String>) -> Option<String> {
    s.filter(|x| !x.is_empty())
}

fn optional_threads(t: Option<u32>) -> Option<u32> {
    t.filter(|&x| x != 0)
}

fn optional_nonempty_string(s: Option<String>, default: &str) -> String {
    s.filter(|x| !x.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn optional_outdir(p: Option<PathBuf>) -> PathBuf {
    p.filter(|x| !x.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Print a commented example configuration file to stdout.
pub fn print_default_config() {
    println!("{}", DEFAULT_CONFIG_TOML);
}

const DEFAULT_CONFIG_TOML: &str = r#"# Cosyn TOML configuration example
#
# Run with:
#   cosyn run-config -c config.toml
#
# Only the section matching `command` is read; all other sections are ignored.

# Subcommand to run. Supported values:
#   fit, ufit, cai, mfe, gc, palindrome, mutate2stop, snp, assay, pll, explore
command = "fit"

# ============================================================================
# fit  -- optimize a CDS/RNA/amino-acid sequence
# ============================================================================
[fit]
fasta = "input.fa"              # required: input FASTA file
freq_table_info = "hc"          # codon frequency table short name
win_size = 50                   # beam search window size
rnd_size = 2                    # random walk size
num_outputs = 1                 # number of candidate sequences to output
lambda = 3.0                    # balance parameter between MFE and CAI
threads = 0                     # number of threads (0 means use all CPUs)
seq_parallel = 1                # number of sequences processed simultaneously
outdir = "."                    # output directory
prefix = ""                     # output file prefix (empty means no prefix)
is_aa_seq = false               # input is amino acid sequences
is_rna = false                  # input is RNA sequences (U instead of T)
homopolymers = 6                # max allowed homopolymer length (0 disables)
avoid_seqs = ""                 # optional file with motifs to avoid

# Optimization objective switches (only one can be true at a time)
gc_cai = false                  # optimize GC + CAI only (ignore MFE)
mfe_cai = false                 # optimize MFE + CAI only (ignore GC)
cmgmp = false                   # MFE/CAI/GC/mutate2stop/palindrome objective
pmgc = false                    # CAI/GC/mutate2stop/palindrome objective
palindrome = false              # add palindrome term to the default objective
maximize_loss = false           # deoptimization mode (maximize loss)
no_verbose = false              # suppress progress bars

# Objective weights (used by cmgmp/pmgc and palindrome modes)
weight_gc = 1.0
weight_cai = 1.0
weight_m2s = 1.0
weight_palindrome = 1.0

# MFE backend: "rust" (pure-Rust LinearFold, default) or "cpp" (deprecated alias)
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)
mfe_pll_start_codon = 0         # codon index where MFE/PLL evaluation starts

# CAI variant used in the objective: "scaled", "arithmetic", or "geometric"
cai_mode = "scaled"

# Search method: "beam" or "mcmc"
search_method = "beam"
mcmc_iterations = 1000
mcmc_mutations = 5
mcmc_accept_prob = 0.5
mcmc_temperature = 0.0
mcmc_traces = 1
mcmc_region = ""                # e.g. "30-90" (base positions, multiples of 3)
weak_head = 0                   # 5'-end bases to keep unstructured (0=off, must be multiple of 3)

# ============================================================================
# ufit -- unified objective optimization
# ============================================================================
[ufit]
fasta = "input.fa"
freq_table_info = "hc"
win_size = 50
rnd_size = 2
num_outputs = 1
lambda = 3.0
weight_gc = 1.0
weight_cai = 1.0
weight_m2s = 1.0
weight_palindrome = 1.0
weight_mfe = 1                  # MFE weight in the unified objective
weight_pll = 0.0                # CodonTransformer PLL weight
model_path = ""                 # required for PLL (CodonTransformer JIT model)
threads = 0
seq_parallel = 1
outdir = "."
prefix = ""
is_aa_seq = false
is_rna = false
homopolymers = 6
avoid_seqs = ""
maximize_loss = false
no_verbose = false
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)
mfe_pll_start_codon = 0

# CAI variant used in the unified objective: "arithmetic", "scaled", or "geometric"
cai_mode = "arithmetic"

search_method = "beam"
mcmc_iterations = 1000
mcmc_mutations = 5
mcmc_accept_prob = 0.5
mcmc_temperature = 0.0
mcmc_traces = 1
mcmc_region = ""
weak_head = 0                   # 5'-end bases to keep unstructured (0=off)

# PLL GPU batch tuning (only used when model_path is set)
pll_batch_size = 256            # max sequences per GPU batch
pll_timeout_ms = 50             # batch accumulation timeout (ms)
pll_threads = 0                 # PLL threads (0 = auto, ≥32 when PLL enabled)

# ============================================================================
# cai  -- calculate Codon Adaptation Index
# ============================================================================
[cai]
fasta = "input.fa"
freq_table_info = "hc"
outdir = "."
prefix = ""

# ============================================================================
# mfe  -- calculate Minimum Free Energy
# ============================================================================
[mfe]
fasta = "input.fa"
outdir = "."
prefix = ""
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)

# ============================================================================
# gc  -- calculate GC content and theoretical bounds
# ============================================================================
[gc]
fasta = "input.fa"
outdir = "."
prefix = ""

# ============================================================================
# palindrome  -- detect palindromic subsequences
# ============================================================================
[palindrome]
fasta = "input.fa"
outdir = "."
prefix = ""

# ============================================================================
# mutate2stop  -- count single-nucleotide mutations to stop codons
# ============================================================================
[mutate2stop]
fasta = "input.fa"
outdir = "."
prefix = ""

# ============================================================================
# snp  -- optimize while fixing user-specified amino-acid mutations
# ============================================================================
[snp]
fasta = "input.fa"
mutations = ["A4R"]             # required: list of mutations like "A4R"
freq_table_info = "hc"
lambda = 2.0
threads = 0
homopolymers = 6
outdir = "."
prefix = ""
avoid_seqs = ""
gc_cai = false
mfe_cai = false
palindrome = false
cmgmp = false
pmgc = false
weight_gc = 1.0
weight_cai = 1.0
weight_palindrome = 1.0
weight_m2s = 1.0
maximize_loss = false
no_verbose = false
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)
mfe_pll_start_codon = 0
search_method = "beam"
mcmc_iterations = 1000
mcmc_mutations = 5
mcmc_accept_prob = 0.5
mcmc_temperature = 0.0
mcmc_traces = 1
mcmc_region = ""

# ============================================================================
# assay  -- evaluate palindrome / MFE / GC / mutate2stop / CAI (and optional PLL)
# ============================================================================
[assay]
fasta = "input.fa"
freq_table_info = "hc"
outdir = "."
prefix = ""
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)
model_path = ""                 # optional: CodonTransformer JIT model; empty = skip PLL

# ============================================================================
# pll  -- evaluate CodonTransformer pseudo-log-likelihood
# ============================================================================
[pll]
fasta = "input.fa"
model_path = "codon_transformer_human.pt"  # required: JIT model path
outdir = "."
prefix = ""

# ============================================================================
# explore  -- panoramic design exploration (optimized / intermediate / deoptimized)
# ============================================================================
[explore]
fasta = "input.fa"
codon_table = "hc"
outdir = "explore_out"
prefix = ""
mfe_method = "rust"
mfe_window = 100                # maximum base-pair span (0 = unlimited, default: 100)
threads = 0                     # number of threads (0 means use all CPUs)
extra_tables = ""               # comma-separated extra codon tables (e.g. "ecoli,li")
no_opt = false                  # skip optimized designs
no_intermediate = false         # skip intermediate designs
no_deopt = false                # skip deoptimized designs
skip_mfe = false                # skip MFE calculation entirely (much faster)
pll_weight = 0.0                # PLL weight for optional PLL-guided MCMC mode
pll_mcmc_iters = 100            # MCMC iterations for PLL-guided intermediate mode
model_path = ""                 # required if pll_weight > 0
intermediate_iters = "100,300,1000"
win_size = 50
lambda = 3.0
homopolymers = 6
avoid_seqs = ""
is_aa = false
is_rna = false
no_verbose = false
"#;

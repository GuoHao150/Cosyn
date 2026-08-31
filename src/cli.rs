use crate::cai::CaiMode;
use crate::mfe_binding::MfeMethod;
use serde::Serialize;
use std::path::PathBuf;
use structopt::StructOpt;

const FREQ_TABLE_INFO: &'static str = r###"""
Codon frequency table: either a built-in short name (listed below)
or a path to a custom .toml file (see example/codon_table_template.toml).

Built-in short names:

bcl, bsc, ems, bhp, bfc, csto, cpla, lu, th, cpit, ac, v, bna, ctes, cspl, 
ct, hlv, bsn, cthy, ceso, bam, cliv, li, cgal, sit, chea, msg, csal, pr, 
cski, ccol, cbre, cart, cmus, pi, ms, bce, as, egj, cfal, wb, ema, ty, av,
cvag, bch, cec, cs, bhi, cner, s, bp, bac, kc, cute, csmo, ckid, cton, cduo, 
cen, cova, cbra, clun, cpro, ag, cfat, hc, cadr, curi, sp, ov, ne, csma, ft, 
at, pa, u, haa, sns, b, aa, capp, cpan, crec, bc, clym, sse, ecoli

the follwing contents is the meaning of these short names, Note, if you select CUSTOM database you must remove the stop codon in your input file.

    hc    -> HumanCommon
    as    -> AdiposeSubcutaneous
    ms    -> MuscleSkeletal
    at    -> ArteryTibial
    ac    -> ArteryCoronary
    haa   -> HeartAtrialAppendage
    av    -> AdiposeVisceral
    u     -> Uterus
    v     -> Vagina
    b     -> BreastMammaryTissue
    sns   -> SkinNotSunExposed
    sse   -> SkinSunExposed
    msg   -> MinorSalivaryGland
    bc    -> BrainCortex
    bcl   -> BrainCerebellum
    bfc   -> BrainFrontalCortex
    bce   -> BrainCaudate
    bna   -> BrainNucleusAccumbens
    bp    -> BrainPutamen
    bhp   -> BrainHypothalamus
    bsc   -> BrainSpinalCord
    bhi   -> BrainHippocampus
    bsn   -> BrainSubstantiaNigra
    bac   -> BrainAnteriorCingulateCortex
    bam   -> BrainAmygdala
    bch   -> BrainCerebellarHemisphere
    ov    -> Ovary
    ag    -> AdrenalGland
    th    -> Thyroid
    lu    -> Lung
    li    -> Liver
    kc    -> KidneyCortex
    sp    -> Spleen
    pa    -> Pancreas
    ems   -> EsophagusMuscularis
    ema   -> EsophagusMucosa
    egj   -> EsophagusGastroesophagealJunction
    s     -> Stomach
    cs    -> ColonSigmoid
    sit   -> SmallIntestineTerminalIleum
    ct    -> ColonTransverse
    pr    -> Prostate
    ty    -> Testis
    ne    -> NerveTibial
    hlv   -> HeartLeftVentricle
    wb    -> WholeBlood
    aa    -> ArteryAorta
    pi    -> Pituitary
    cec   -> CervixEctocervix
    ft    -> FallopianTube
    cen   -> CervixEndocervix
    clun  -> Custom_LUNG
    cbre  -> Custom_BREAST
    cski  -> Custom_SKIN
    cspl  -> Custom_SPLEEN
    chea  -> Custom_HEART
    cliv  -> Custom_LIVER
    csal  -> Custom_SALIVARYGLAND
    cmus  -> Custom_MUSCLE_SKELETAL
    cton  -> Custom_TONSIL
    csma  -> Custom_SMALLINTESTINE
    cpla  -> Custom_PLACENTA
    capp  -> Custom_APPENDICES
    ctes  -> Custom_TESTIS
    crec  -> Custom_RECTUM
    curi  -> Custom_URINARY_BLADDER
    cpro  -> Custom_PROSTATE
    ceso  -> Custom_ESOPHAGUS
    ckid  -> Custom_KIDNEY
    cthy  -> Custom_THYROID
    clym  -> Custom_LYMPHNODE
    cart  -> Custom_ARTERY
    cbra  -> Custom_BRAIN
    cner  -> Custom_NERVE_TIBIAL
    cgal  -> Custom_GALLBLADDER
    cute  -> Custom_UTERUS
    cpit  -> Custom_PITUITARY
    ccol  -> Custom_COLON
    cvag  -> Custom_VAGINA
    cduo  -> Custom_DUODENUM
    cfat  -> Custom_FAT
    csto  -> Custom_STOMACH
    cadr  -> Custom_ADRENAL
    cfal  -> Custom_FALLOPIANTUBE
    csmo  -> Custom_SMOOTHMUSCLE
    cpan  -> Custom_PANCREAS
    cova  -> Custom_OVARY
    ecoli -> EColi
"""###;

const TOP_INFO: &'static str = r#"
Cosyn is a command-line tool for optimizing/deoptimizing the codon sequences.
Six indicators are used to evaluate the codon sequences:
1. GC%: the GC content of the codon sequences
2. CAI: the CAI of the codon sequences
3. mfe: the MFE of the codon sequences
4. palindrom_score: palindromic sequence number
5. mutate2stop_score: how easily those codons will mutate to stop codons
6. codon_pll: Pseudo log-likelihood value for the codon sequence calculated by the CodonTransformer 
"#;

#[derive(Debug, StructOpt, Serialize)]
#[structopt(
    name = "Cosyn",
    about = TOP_INFO,
)]
pub enum SubCommand {
    #[structopt(name = "fit", about = "Optimize the input DNA/RNA/AA sequences")]
    Fit(FitOptions),

    #[structopt(name = "ufit", about = "Optimize the input DNA/RNA/AA sequences")]
    Ufit(UfitOptions),

    #[structopt(name = "cai", about = "Calculate the CAI of the input CDS sequences")]
    Cai(CaiOptions),

    #[structopt(name = "mfe", about = "Calculate the MFE of the input CDS sequences")]
    Mfe(MfeOptions),

    #[structopt(
        name = "gc",
        about = "Calculate the GC of input DNA sequences and also their theoretical min/max GC after optimization"
    )]
    Gc(GcOptions),

    #[structopt(
        name = "palindrome",
        about = "To find the palindromic sequences in the input DNA sequences"
    )]
    Palindrome(PalindromeOptions),

    #[structopt(
        name = "mutate2stop",
        about = "Calculate the raw mutate2stop numbers of the input DNA sequences"
    )]
    Mutate2Stop(M2sOptions),

    #[structopt(
        name = "snp",
        about = "Partially Optimizing the input CDS sequence with the target mutation in its amino acids sequence while not changing the rest nucleotides in the CDS"
    )]
    SnpFit(SnpOptions),

    #[structopt(
        name = "assay",
        about = "Evaluating the MFE GC% CAI palindrome-seqs and mutate2stop score on the input DNA sequences"
    )]
    Assay(AssayOptions),

    #[structopt(
        name = "pll",
        about = "Calculate the codon PLL of the input CDS sequences using a CodonTransformer model"
    )]
    CodonPll(CodonPllOptions),

    #[structopt(
        name = "run-config",
        about = "Run cosyn from a TOML configuration file"
    )]
    RunConfig(RunConfigOptions),

    #[structopt(
        name = "explore",
        about = "Automatically generate a diverse set of optimized, intermediate, and deoptimized designs"
    )]
    Explore(ExploreOptions),
}

#[derive(Debug, StructOpt, Serialize)]
pub struct RunConfigOptions {
    /// Path to the TOML configuration file. Required unless --print-default-config is used.
    #[structopt(short = "c", long = "config", parse(from_os_str))]
    pub config: Option<PathBuf>,

    /// Print a commented example TOML configuration to stdout and exit.
    #[structopt(long = "print-default-config")]
    pub print_default_config: bool,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct ExploreOptions {
    /// Input FASTA file (DNA/RNA/AA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,

    #[structopt(short = "c", long = "codon_table", help = FREQ_TABLE_INFO)]
    pub codon_table: String,

    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = "explore_out")]
    pub outdir: PathBuf,

    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,

    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "100",
        help = "Maximum base-pair span for MFE folding (0 = unlimited, default: 100)"
    )]
    pub mfe_window: usize,

    /// Number of threads (default: all CPUs).
    #[structopt(short = "t", long = "threads")]
    pub threads: Option<u32>,
    /// Comma-separated extra codon tables (e.g. `hc,ecoli`). The same design schedule is run for each table.
    #[structopt(long = "extra-tables")]
    pub extra_tables: Option<String>,

    /// Skip the optimized-design zone.
    #[structopt(long = "no-opt")]
    pub no_opt: bool,

    /// Skip the intermediate-design zone.
    #[structopt(long = "no-intermediate")]
    pub no_intermediate: bool,

    /// Skip the deoptimized-design zone.
    #[structopt(long = "no-deopt")]
    pub no_deopt: bool,

    /// Skip MFE calculation entirely during design (sets --mfe_pll_start to a very large value).
    #[structopt(long = "skip-mfe")]
    pub skip_mfe: bool,

    /// PLL weight for an additional PLL-guided intermediate MCMC mode (requires --model-path).
    #[structopt(long = "pll-weight", default_value = "0.0")]
    pub pll_weight: f64,

    /// MCMC iterations for the optional PLL-guided intermediate mode.
    #[structopt(long = "pll-mcmc-iters", default_value = "100")]
    pub pll_mcmc_iters: usize,

    /// Path to a JIT-traced CodonTransformer model for --pll-weight.
    #[structopt(long = "model-path", parse(from_os_str))]
    pub model_path: Option<PathBuf>,

    /// Comma-separated MCMC iteration counts for intermediate designs.
    #[structopt(long = "intermediate-iters", default_value = "100,300,1000")]
    pub intermediate_iters: String,

    /// Beam search window size (larger = better quality, slower).
    #[structopt(short = "w", long = "win_size", default_value = "50")]
    pub win_size: u32,
    /// Balance parameter for MFE/CAI trade-off (higher = stronger MFE weight).
    #[structopt(short = "l", long = "lambda", default_value = "3")]
    pub lambda: f64,
    /// Maximum allowed homopolymer run length.
    #[structopt(short = "y", long = "homopolymers", default_value = "6")]
    pub homopolymers: usize,
    /// FASTA file of motifs to avoid in output sequences.
    #[structopt(short = "a", long = "avoid_seqs")]
    pub avoid_seqs: Option<PathBuf>,
    /// Input is amino acid sequences (will be reverse-translated to CDS).
    #[structopt(long = "is_aa")]
    pub is_aa: bool,
    /// Input is RNA sequences (U instead of T).
    #[structopt(long = "is_rna")]
    pub is_rna: bool,
    /// Suppress verbose progress output.
    #[structopt(long = "no_verbose")]
    pub no_verbose: bool,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct FitOptions {
    /// Input FASTA file (DNA/RNA/AA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Beam search window size (larger = better quality, slower).
    #[structopt(short = "w", long = "win_size", default_value = "50")]
    pub win_size: u32,
    /// Random walk size for beam search exploration.
    #[structopt(short = "r", long = "rnd_size", default_value = "2")]
    pub rnd_size: u32,
    /// Number of candidate sequences to output.
    #[structopt(short = "n", long = "num_outputs", default_value = "1")]
    pub num_outputs: u32,
    /// Balance parameter for MFE/CAI trade-off (higher = stronger MFE weight).
    #[structopt(short = "l", long = "lambda", default_value = "3")]
    pub lambda: f64,
    /// Number of threads (default: all CPUs).
    #[structopt(short = "t", long = "threads")]
    pub threads: Option<u32>,
    /// Number of sequences to process in parallel.
    #[structopt(short = "s", long = "seq_parallel", default_value = "1")]
    pub seq_parallel: u32,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Input is amino acid sequences (will be reverse-translated to CDS).
    #[structopt(long = "is_aa")]
    pub is_aa_seq: bool,
    /// Input is RNA sequences (U instead of T).
    #[structopt(long = "is_rna")]
    pub is_rna: bool,
    /// Maximum allowed homopolymer run length.
    #[structopt(short = "y", long = "homopolymers", default_value = "6")]
    pub homopolymers: usize,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    /// FASTA file of motifs to avoid in output sequences.
    #[structopt(short = "a", long = "avoid_seqs")]
    pub avoid_seqs: Option<PathBuf>,
    #[structopt(short = "c", long = "codon_table", help=FREQ_TABLE_INFO)]
    pub freq_table_info: String,
    /// Optimize GC content and CAI only (skip MFE calculation, fastest).
    #[structopt(long = "gc_cai")]
    pub gc_cai: bool,
    /// Optimize MFE and CAI only (skip GC content).
    #[structopt(long = "mfe_cai")]
    pub mfe_cai: bool,
    /// Suppress verbose progress output.
    #[structopt(long = "no_verbose")]
    pub no_verbose: bool,
    /// Include palindrome avoidance in the optimization objective.
    #[structopt(long = "palindrome")]
    pub palindrome: bool,
    /// Optimize all five objectives: CAI + MFE + GC + M2S + Palindrome.
    #[structopt(long = "cmgmp")]
    pub cmgmp: bool,
    /// Optimize Palindrome + M2S + GC + CAI (omit MFE for speed).
    #[structopt(long = "pmgc")]
    pub pmgc: bool,
    /// Weight for GC content in the loss function.
    #[structopt(
        short = "b",
        long = "gc_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_gc: f64,
    /// Weight for CAI in the loss function.
    #[structopt(
        short = "x",
        long = "cai_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_cai: f64,
    /// Weight for mutate-to-stop score in the loss function.
    #[structopt(
        short = "d",
        long = "m2s_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_m2s: f64,
    /// Weight for palindrome score in the loss function.
    #[structopt(
        short = "m",
        long = "palindrome_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_palindrome: f64,
    /// Deoptimization mode: maximize loss instead of minimize (for attenuated vaccine design).
    #[structopt(long = "maximize_loss")]
    pub maximize_loss: bool,
    /// Codon position (0-based) to start MFE/PLL calculation (0 = full sequence).
    #[structopt(short = "k", long = "mfe_pll_start", default_value = "0")]
    pub mfe_pll_start_codon: usize,
    #[structopt(
        long = "search-method",
        default_value = "beam",
        help = "Optimization algorithm: beam or mcmc"
    )]
    pub search_method: String,
    #[structopt(
        long = "mcmc-iterations",
        default_value = "1000",
        help = "Number of MCMC iterations"
    )]
    pub mcmc_iterations: usize,
    #[structopt(
        long = "mcmc-mutations",
        default_value = "5",
        help = "Number of codons mutated per MCMC proposal"
    )]
    pub mcmc_mutations: usize,
    #[structopt(
        long = "mcmc-accept-prob",
        default_value = "0.5",
        help = "Fixed acceptance probability for worse MCMC proposals"
    )]
    pub mcmc_accept_prob: f64,
    #[structopt(
        long = "mcmc-temperature",
        default_value = "0.0",
        help = "Boltzmann temperature for MCMC (0 disables Boltzmann)"
    )]
    pub mcmc_temperature: f64,
    #[structopt(
        long = "mcmc-traces",
        default_value = "1",
        help = "Number of independent MCMC chains to run"
    )]
    pub mcmc_traces: usize,
    #[structopt(
        long = "mcmc-region",
        help = "Optimize only a CDS region, given as base positions start-end (e.g. 30-90). Must be multiples of 3; requires DNA/RNA input."
    )]
    pub mcmc_region: Option<String>,
    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "100",
        help = "Maximum base-pair span for MFE folding (0 = unlimited, default: 100)"
    )]
    pub mfe_window: usize,
    #[structopt(
        long = "weak-head",
        default_value = "0",
        help = "Number of 5'-end bases (nt) to keep unstructured. Must be a multiple of 3. \
                When > 0, avoids stable RNA secondary structure (hairpins) near the start codon. \
                0 = disabled."
    )]
    pub weak_head: usize,
    #[structopt(
        long = "cai-mode",
        default_value = "scaled",
        help = "CAI variant used in the objective: scaled (LinearDesign-style), arithmetic, or geometric."
    )]
    pub cai_mode: CaiMode,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct UfitOptions {
    /// Input FASTA file (DNA/RNA/AA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Beam search window size (larger = better quality, slower).
    #[structopt(short = "w", long = "win_size", default_value = "50")]
    pub win_size: u32,
    /// Random walk size for beam search exploration.
    #[structopt(short = "r", long = "rnd_size", default_value = "2")]
    pub rnd_size: u32,
    /// Number of candidate sequences to output.
    #[structopt(short = "n", long = "num_outputs", default_value = "1")]
    pub num_outputs: u32,
    /// Balance parameter for MFE/CAI trade-off (higher = stronger MFE weight).
    #[structopt(short = "l", long = "lambda", default_value = "3")]
    pub lambda: f64,
    /// Weight for GC content in the unified loss function.
    #[structopt(
        short = "b",
        long = "gc_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_gc: f64,
    /// Weight for CAI in the unified loss function.
    #[structopt(
        short = "x",
        long = "cai_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_cai: f64,
    /// Weight for mutate-to-stop score in the unified loss function.
    #[structopt(
        short = "d",
        long = "m2s_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_m2s: f64,
    /// Weight for palindrome score in the unified loss function.
    #[structopt(
        short = "m",
        long = "palindrome_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_palindrome: f64,
    /// Weight for MFE in the unified loss function.
    #[structopt(
        short = "e",
        long = "mfe_weight",
        default_value = "1",
        allow_hyphen_values = true
    )]
    pub weight_mfe: i64,
    /// Weight for CodonTransformer PLL in the unified loss function (requires --model-path).
    #[structopt(
        short = "q",
        long = "pll_weight",
        default_value = "0.0",
        allow_hyphen_values = true
    )]
    pub weight_pll: f64,
    /// Path to a JIT-traced CodonTransformer model for PLL guidance.
    #[structopt(short = "j", long = "model_path")]
    pub model_path: Option<PathBuf>,
    /// Number of threads (default: all CPUs).
    #[structopt(short = "t", long = "threads")]
    pub threads: Option<u32>,
    /// Number of sequences to process in parallel.
    #[structopt(short = "s", long = "seq_parallel", default_value = "1")]
    pub seq_parallel: u32,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Input is amino acid sequences (will be reverse-translated to CDS).
    #[structopt(long = "is_aa")]
    pub is_aa_seq: bool,
    /// Input is RNA sequences (U instead of T).
    #[structopt(long = "is_rna")]
    pub is_rna: bool,
    /// Maximum allowed homopolymer run length.
    #[structopt(short = "y", long = "homopolymers", default_value = "6")]
    pub homopolymers: usize,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    /// FASTA file of motifs to avoid in output sequences.
    #[structopt(short = "a", long = "avoid_seqs")]
    pub avoid_seqs: Option<PathBuf>,
    #[structopt(short = "c", long = "codon_table", help=FREQ_TABLE_INFO)]
    pub freq_table_info: String,
    /// Deoptimization mode: maximize loss instead of minimize (for attenuated vaccine design).
    #[structopt(long = "maximize_loss")]
    pub maximize_loss: bool,
    /// Suppress verbose progress output.
    #[structopt(long = "no_verbose")]
    pub no_verbose: bool,
    /// Codon position (0-based) to start MFE/PLL calculation (0 = full sequence).
    #[structopt(short = "k", long = "mfe_pll_start", default_value = "0")]
    pub mfe_pll_start_codon: usize,
    #[structopt(
        long = "search-method",
        default_value = "beam",
        help = "Optimization algorithm: beam or mcmc"
    )]
    pub search_method: String,
    #[structopt(
        long = "mcmc-iterations",
        default_value = "1000",
        help = "Number of MCMC iterations"
    )]
    pub mcmc_iterations: usize,
    #[structopt(
        long = "mcmc-mutations",
        default_value = "5",
        help = "Number of codons mutated per MCMC proposal"
    )]
    pub mcmc_mutations: usize,
    #[structopt(
        long = "mcmc-accept-prob",
        default_value = "0.5",
        help = "Fixed acceptance probability for worse MCMC proposals"
    )]
    pub mcmc_accept_prob: f64,
    #[structopt(
        long = "mcmc-temperature",
        default_value = "0.0",
        help = "Boltzmann temperature for MCMC (0 disables Boltzmann)"
    )]
    pub mcmc_temperature: f64,
    #[structopt(
        long = "mcmc-traces",
        default_value = "1",
        help = "Number of independent MCMC chains to run"
    )]
    pub mcmc_traces: usize,
    #[structopt(
        long = "mcmc-region",
        help = "Optimize only a CDS region, given as base positions start-end (e.g. 30-90). Must be multiples of 3; requires DNA/RNA input."
    )]
    pub mcmc_region: Option<String>,
    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "100",
        help = "Maximum base-pair span for MFE folding (0 = unlimited, default: 100)"
    )]
    pub mfe_window: usize,
    #[structopt(
        long = "weak-head",
        default_value = "0",
        help = "Number of 5'-end bases (nt) to keep unstructured. Must be a multiple of 3. \
                When > 0, avoids stable RNA secondary structure (hairpins) near the start codon. \
                0 = disabled."
    )]
    pub weak_head: usize,
    #[structopt(
        long = "cai-mode",
        default_value = "arithmetic",
        help = "CAI variant used in the unified objective: arithmetic, scaled, or geometric."
    )]
    pub cai_mode: CaiMode,
    #[structopt(
        long = "pll-batch-size",
        default_value = "256",
        help = "Max sequences per GPU PLL inference batch. Larger values improve GPU \
                utilization but use more VRAM. Only used when --model-path is set."
    )]
    pub pll_batch_size: usize,
    #[structopt(
        long = "pll-timeout-ms",
        default_value = "50",
        help = "Max milliseconds the PLL batch service waits to accumulate a batch. \
                Longer waits yield bigger batches (better GPU util) but add latency. \
                Only used when --model-path is set."
    )]
    pub pll_timeout_ms: usize,
    #[structopt(
        long = "pll-threads",
        default_value = "0",
        help = "Number of CPU threads used for PLL loss computation. \
                0 = auto (min 32 when PLL is enabled, otherwise num_cpus). \
                More threads feed the GPU batch more efficiently."
    )]
    pub pll_threads: usize,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct CaiOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    #[structopt(short = "c", long = "codon_table", help=FREQ_TABLE_INFO)]
    pub freq_table_info: String,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct MfeOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "0",
        help = "Maximum base-pair span for MFE folding (0 = unlimited)"
    )]
    pub mfe_window: usize,
    /// Number of threads (default: all CPUs).
    #[structopt(short = "t", long = "threads")]
    pub threads: Option<u32>,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct GcOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct PalindromeOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct AssayOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    #[structopt(short = "c", long = "codon_table", help=FREQ_TABLE_INFO)]
    pub freq_table_info: String,
    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "0",
        help = "Maximum base-pair span for MFE folding (0 = unlimited)"
    )]
    pub mfe_window: usize,
    #[structopt(
        short = "j",
        long = "model_path",
        help = "Optional CodonTransformer JIT model path; if provided, PLL is computed alongside other assay metrics"
    )]
    pub model_path: Option<PathBuf>,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct CodonPllOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Path to a JIT-traced CodonTransformer model.
    #[structopt(short = "j", long = "model_path")]
    pub model_path: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct SnpOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    #[structopt(
        short = "s",
        long = "snp_sites",
        required = true,
        min_values = 1,
        parse(from_str)
    )]
    pub mutations: Vec<String>,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Balance parameter for MFE/CAI trade-off (higher = stronger MFE weight).
    #[structopt(short = "l", long = "lambda", default_value = "2")]
    pub lambda: f64,
    /// Number of threads (default: all CPUs).
    #[structopt(short = "t", long = "threads")]
    pub threads: Option<u32>,
    /// Maximum allowed homopolymer run length.
    #[structopt(short = "y", long = "homopolymers", default_value = "6")]
    pub homopolymers: usize,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
    /// FASTA file of motifs to avoid in output sequences.
    #[structopt(short = "a", long = "avoid_seqs")]
    pub avoid_seqs: Option<PathBuf>,
    #[structopt(short = "c", long = "codon_table", help=FREQ_TABLE_INFO)]
    pub freq_table_info: String,
    /// Optimize GC content and CAI only (skip MFE calculation, fastest).
    #[structopt(long = "gc_cai")]
    pub gc_cai: bool,
    /// Optimize MFE and CAI only (skip GC content).
    #[structopt(long = "mfe_cai")]
    pub mfe_cai: bool,
    /// Include palindrome avoidance in the optimization objective.
    #[structopt(long = "palindrome")]
    pub palindrome: bool,
    /// Optimize all five objectives: CAI + MFE + GC + M2S + Palindrome.
    #[structopt(long = "cmgmp")]
    pub cmgmp: bool,
    /// Optimize Palindrome + M2S + GC + CAI (omit MFE for speed).
    #[structopt(long = "pmgc")]
    pub pmgc: bool,
    /// Weight for GC content in the loss function.
    #[structopt(
        short = "b",
        long = "gc_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_gc: f64,
    /// Weight for CAI in the loss function.
    #[structopt(
        short = "x",
        long = "cai_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_cai: f64,
    /// Weight for palindrome score in the loss function.
    #[structopt(
        short = "m",
        long = "palindrome_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_palindrome: f64,
    /// Weight for mutate-to-stop score in the loss function.
    #[structopt(
        short = "d",
        long = "m2s_weight",
        default_value = "1.0",
        allow_hyphen_values = true
    )]
    pub weight_m2s: f64,
    /// Deoptimization mode: maximize loss instead of minimize (for attenuated vaccine design).
    #[structopt(long = "maximize_loss")]
    pub maximize_loss: bool,
    /// Suppress verbose progress output.
    #[structopt(long = "no_verbose")]
    pub no_verbose: bool,
    /// Codon position (0-based) to start MFE/PLL calculation (0 = full sequence).
    #[structopt(short = "k", long = "mfe_pll_start", default_value = "0")]
    pub mfe_pll_start_codon: usize,
    #[structopt(
        long = "search-method",
        default_value = "beam",
        help = "Optimization algorithm: beam or mcmc"
    )]
    pub search_method: String,
    #[structopt(
        long = "mcmc-iterations",
        default_value = "1000",
        help = "Number of MCMC iterations"
    )]
    pub mcmc_iterations: usize,
    #[structopt(
        long = "mcmc-mutations",
        default_value = "5",
        help = "Number of codons mutated per MCMC proposal"
    )]
    pub mcmc_mutations: usize,
    #[structopt(
        long = "mcmc-accept-prob",
        default_value = "0.5",
        help = "Fixed acceptance probability for worse MCMC proposals"
    )]
    pub mcmc_accept_prob: f64,
    #[structopt(
        long = "mcmc-temperature",
        default_value = "0.0",
        help = "Boltzmann temperature for MCMC (0 disables Boltzmann)"
    )]
    pub mcmc_temperature: f64,
    #[structopt(
        long = "mcmc-traces",
        default_value = "1",
        help = "Number of independent MCMC chains to run"
    )]
    pub mcmc_traces: usize,
    #[structopt(
        long = "mcmc-region",
        help = "Optimize only a CDS region, given as base positions start-end (e.g. 30-90). Must be multiples of 3; requires DNA/RNA input."
    )]
    pub mcmc_region: Option<String>,
    #[structopt(skip)]
    pub mfe_method: MfeMethod,
    #[structopt(
        long = "mfe-window",
        default_value = "100",
        help = "Maximum base-pair span for MFE folding (0 = unlimited, default: 100)"
    )]
    pub mfe_window: usize,
}

#[derive(Debug, StructOpt, Serialize)]
pub struct M2sOptions {
    /// Input FASTA file (DNA sequences).
    #[structopt(short = "f", long = "fasta")]
    pub fasta: PathBuf,
    /// Output directory.
    #[structopt(short = "o", long = "outdir", default_value = ".")]
    pub outdir: PathBuf,
    /// Output file prefix.
    #[structopt(short = "p", long = "prefix")]
    pub prefix: Option<String>,
}

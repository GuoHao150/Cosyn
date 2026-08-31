use crate::cai::*;
use crate::cds::{CDSSeq, LossFuncs, LossInfo, Triplet};
use crate::mfe_binding::{call_mfe, MfeMethod};
use crate::mutate2stop::mutate2stop_numbers;
use crate::palindrome::palindrome_score;
use crate::AvoidSeqs;
use call_mfe::compute_mfe;
use itertools::{Itertools, MultiProduct};
use kdam::{tqdm, BarExt};
use rand::prelude::*;
use rand::seq::IteratorRandom;
use rand::{self, Rng};
use serde::Serialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::u64;
use std::vec::IntoIter;
use threadpool::ThreadPool;

const RND_WALK_THRESHOLD: f64 = 0.95;
const DISPLAY_STR_LEN: usize = 32;
const SEND_ERR_MEG: &'static str = "failed to send loss updated SearchPath";

#[derive(Clone, Serialize)]
pub enum OptimizeObject {
    GcCai,
    MfeCai,
    MfeGcCai,
    GcCaiPalin,
    MfeCaiPalin,
    MfeGcCaiPalin,
    // Deoptimization.
    // cmpmp
    MfeCaiGcM2sPalin,
    // pmgc
    CaiGcM2sPalin,
    UnifiedObj,
}

/// The enum represents the codon frequency tables
#[derive(Clone, Serialize)]
pub enum CodonTables {
    Common,
    AdiposeSubcutaneous,
    MuscleSkeletal,
    ArteryTibial,
    ArteryCoronary,
    HeartAtrialAppendage,
    AdiposeVisceral,
    Uterus,
    Vagina,
    BreastMammaryTissue,
    SkinNotSunExposed,
    MinorSalivaryGland,
    BrainCortex,
    AdrenalGland,
    Thyroid,
    Lung,
    Spleen,
    Pancreas,
    EsophagusMuscularis,
    EsophagusMucosa,
    EsophagusGastroesophagealJunction,
    Stomach,
    ColonSigmoid,
    SmallIntestineTerminalIleum,
    ColonTransverse,
    Prostate,
    Testis,
    NerveTibial,
    SkinSunExposed,
    HeartLeftVentricle,
    BrainCerebellum,
    WholeBlood,
    ArteryAorta,
    Pituitary,
    BrainFrontalCortex,
    BrainCaudate,
    BrainNucleusAccumbens,
    BrainPutamen,
    BrainHypothalamus,
    BrainSpinalCord,
    BrainHippocampus,
    BrainAnteriorCingulateCortex,
    Ovary,
    BrainCerebellarHemisphere,
    Liver,
    BrainSubstantiaNigra,
    KidneyCortex,
    BrainAmygdala,
    CervixEctocervix,
    FallopianTube,
    CervixEndocervix,
    CustomLung,
    CustomBreast,
    CustomSkin,
    CustomSpleen,
    CustomHeart,
    CustomLiver,
    CustomSalivarygland,
    CustomMuscleSkeletal,
    CustomTonsil,
    CustomSmallintestine,
    CustomPlacenta,
    CustomAppendices,
    CustomTestis,
    CustomRectum,
    CustomUrinarybladder,
    CustomProstate,
    CustomEsophagus,
    CustomKidney,
    CustomThyroid,
    CustomLymphnode,
    CustomArtery,
    CustomBrain,
    CustomNerveTibial,
    CustomGallbladder,
    CustomUterus,
    CustomPituitary,
    CustomColon,
    CustomVagina,
    CustomDuodenum,
    CustomFat,
    CustomStomach,
    CustomAdrenal,
    CustomFallopiantube,
    CustomSmoothmuscle,
    CustomPancreas,
    CustomOvary,
    EColi,
    /// User-provided codon frequency table loaded from a TOML file.
    Custom,
}

/// Pointer to a heap-allocated custom codon table.  Set once by
/// `CodonTables::load_custom_table()` and never freed (intentional leak
/// to obtain a `&'static` lifetime).
static mut CUSTOM_TABLE_PTR: *const CodonTable = std::ptr::null();

impl CodonTables {
    /// Load a custom codon frequency table from a TOML file.
    ///
    /// The TOML file should contain sections named by single-letter amino acid
    /// codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
    /// plus `"*"` for stop codons.  Each section maps DNA codon triplets to
    /// relative frequencies (0–1).  Frequencies within each amino acid should
    /// sum to 1.0.
    ///
    /// Returns `Ok(())` on success or an error message string.
    pub fn load_custom_table(path: &str) -> Result<(), String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("cannot read '{}': {}", path, e))?;

        let raw: toml::Value = content
            .parse()
            .map_err(|e| format!("invalid TOML in '{}': {}", path, e))?;

        let top = raw
            .as_table()
            .ok_or_else(|| "TOML root must be a table (e.g. [A])".to_string())?;

        let mut freqs = [0.0; 64];
        let mut max_freq = [0.0; 21];

        for (aa, codons_val) in top {
            let aa_idx = aa_to_idx(aa);
            let codons = codons_val
                .as_table()
                .ok_or_else(|| format!("[{}] must be a table of codon=frequency pairs", aa))?;

            let mut aa_total = 0.0;
            let mut aa_max = 0.0;
            for (codon, freq_val) in codons {
                let freq = freq_val
                    .as_float()
                    .ok_or_else(|| format!("{}.{} must be a number", aa, codon))?;
                let idx = codon_to_idx(codon.as_bytes());
                freqs[idx as usize] = freq;
                aa_total += freq;
                if freq > aa_max {
                    aa_max = freq;
                }
            }

            // Validate: frequencies within each amino acid should sum to ~1.0
            if (aa_total - 1.0).abs() > 0.05 {
                return Err(format!(
                    "frequencies for amino acid '{}' sum to {:.2} (should be ~1.0)",
                    aa, aa_total
                ));
            }
            max_freq[aa_idx as usize] = aa_max;
        }

        // For amino acids not present in the custom TOML, derive max_freq from
        // the shared genetic code so that synonymous codons still get a ratio.
        for aa_idx in 0..21usize {
            if max_freq[aa_idx] == 0.0 {
                let mut m = 0.0;
                for &idx in CODONS_FOR_AA[aa_idx] {
                    if freqs[idx as usize] > m {
                        m = freqs[idx as usize];
                    }
                }
                max_freq[aa_idx] = m;
            }
        }

        let table = CodonTable { freqs, max_freq };
        let boxed = Box::new(table);
        unsafe {
            CUSTOM_TABLE_PTR = Box::into_raw(boxed);
        }
        Ok(())
    }

    /// Returns true if the argument looks like a TOML file path (ends with .toml).
    pub fn is_toml_path(s: &str) -> bool {
        s.ends_with(".toml")
    }

    /// from the enum get the frequency values
    pub fn get_codon_table(&self) -> &'static CodonTable {
        match self {
            CodonTables::Common => &HUMAN_CODON_TABLE,
            CodonTables::AdiposeSubcutaneous => &HUMAN_ADIPOSE_SUBCUTANEOUS_CODON_TABLE,
            CodonTables::MuscleSkeletal => &HUMAN_MUSCLE_SKELETAL_CODON_TABLE,
            CodonTables::ArteryTibial => &HUMAN_ARTERY_TIBIAL_CODON_TABLE,
            CodonTables::ArteryCoronary => &HUMAN_ARTERY_CORONARY_CODON_TABLE,
            CodonTables::HeartAtrialAppendage => &HUMAN_HEART_ATRIAL_APPENDAGE_CODON_TABLE,
            CodonTables::AdiposeVisceral => &HUMAN_ADIPOSE_VISCERAL_CODON_TABLE,
            CodonTables::Uterus => &HUMAN_UTERUS_CODON_TABLE,
            CodonTables::Vagina => &HUMAN_VAGINA_CODON_TABLE,
            CodonTables::BreastMammaryTissue => &HUMAN_BREAST_MAMMARY_TISSUE_CODON_TABLE,
            CodonTables::SkinNotSunExposed => &HUMAN_SKIN_NOT_SUN_EXPOSED_CODON_TABLE,
            CodonTables::MinorSalivaryGland => &HUMAN_MINOR_SALIVARY_GLAND_CODON_TABLE,
            CodonTables::BrainCortex => &HUMAN_BRAIN_CORTEX_CODON_TABLE,
            CodonTables::AdrenalGland => &HUMAN_ADRENAL_GLAND_CODON_TABLE,
            CodonTables::Thyroid => &HUMAN_THYROID_CODON_TABLE,
            CodonTables::Lung => &HUMAN_LUNG_CODON_TABLE,
            CodonTables::Spleen => &HUMAN_SPLEEN_CODON_TABLE,
            CodonTables::Pancreas => &HUMAN_PANCREAS_CODON_TABLE,
            CodonTables::EsophagusMuscularis => &HUMAN_ESOPHAGUS_MUSCULARIS_CODON_TABLE,
            CodonTables::EsophagusMucosa => &HUMAN_ESOPHAGUS_MUCOSA_CODON_TABLE,
            CodonTables::EsophagusGastroesophagealJunction => {
                &HUMAN_ESOPHAGUS_GASTROESOPHAGEAL_JUNCTION_CODON_TABLE
            }
            CodonTables::Stomach => &HUMAN_STOMACH_CODON_TABLE,
            CodonTables::ColonSigmoid => &HUMAN_COLON_SIGMOID_CODON_TABLE,
            CodonTables::SmallIntestineTerminalIleum => {
                &HUMAN_SMALL_INTESTINE_TERMINAL_ILEUM_CODON_TABLE
            }
            CodonTables::ColonTransverse => &HUMAN_COLON_TRANSVERSE_CODON_TABLE,
            CodonTables::Prostate => &HUMAN_PROSTATE_CODON_TABLE,
            CodonTables::Testis => &HUMAN_TESTIS_CODON_TABLE,
            CodonTables::NerveTibial => &HUMAN_NERVE_TIBIAL_CODON_TABLE,
            CodonTables::SkinSunExposed => &HUMAN_SKIN_SUN_EXPOSED_CODON_TABLE,
            CodonTables::HeartLeftVentricle => &HUMAN_HEART_LEFT_VENTRICLE_CODON_TABLE,
            CodonTables::BrainCerebellum => &HUMAN_BRAIN_CEREBELLUM_CODON_TABLE,
            CodonTables::WholeBlood => &HUMAN_WHOLE_BLOOD_CODON_TABLE,
            CodonTables::ArteryAorta => &HUMAN_ARTERY_AORTA_CODON_TABLE,
            CodonTables::Pituitary => &HUMAN_PITUITARY_CODON_TABLE,
            CodonTables::BrainFrontalCortex => &HUMAN_BRAIN_FRONTAL_CORTEX_CODON_TABLE,
            CodonTables::BrainCaudate => &HUMAN_BRAIN_CAUDATE_CODON_TABLE,
            CodonTables::BrainNucleusAccumbens => &HUMAN_BRAIN_NUCLEUS_ACCUMBENS_CODON_TABLE,
            CodonTables::BrainPutamen => &HUMAN_BRAIN_PUTAMEN_CODON_TABLE,
            CodonTables::BrainHypothalamus => &HUMAN_BRAIN_HYPOTHALAMUS_CODON_TABLE,
            CodonTables::BrainSpinalCord => &HUMAN_BRAIN_SPINAL_CORD_CODON_TABLE,
            CodonTables::BrainHippocampus => &HUMAN_BRAIN_HIPPOCAMPUS_CODON_TABLE,
            CodonTables::BrainAnteriorCingulateCortex => {
                &HUMAN_BRAIN_ANTERIOR_CINGULATE_CORTEX_CODON_TABLE
            }
            CodonTables::Ovary => &HUMAN_OVARY_CODON_TABLE,
            CodonTables::BrainCerebellarHemisphere => {
                &HUMAN_BRAIN_CEREBELLAR_HEMISPHERE_CODON_TABLE
            }
            CodonTables::Liver => &HUMAN_LIVER_CODON_TABLE,
            CodonTables::BrainSubstantiaNigra => &HUMAN_BRAIN_SUBSTANTIA_NIGRA_CODON_TABLE,
            CodonTables::KidneyCortex => &HUMAN_KIDNEY_CORTEX_CODON_TABLE,
            CodonTables::BrainAmygdala => &HUMAN_BRAIN_AMYGDALA_CODON_TABLE,
            CodonTables::CervixEctocervix => &HUMAN_CERVIX_ECTOCERVIX_CODON_TABLE,
            CodonTables::FallopianTube => &HUMAN_FALLOPIAN_TUBE_CODON_TABLE,
            CodonTables::CervixEndocervix => &HUMAN_CERVIX_ENDOCERVIX_CODON_TABLE,
            CodonTables::CustomLung => &HUMAN_CUSTOM_LUNG_CODON_TABLE,
            CodonTables::CustomBreast => &HUMAN_CUSTOM_BREAST_CODON_TABLE,
            CodonTables::CustomSkin => &HUMAN_CUSTOM_SKIN_CODON_TABLE,
            CodonTables::CustomSpleen => &HUMAN_CUSTOM_SPLEEN_CODON_TABLE,
            CodonTables::CustomHeart => &HUMAN_CUSTOM_HEART_CODON_TABLE,
            CodonTables::CustomLiver => &HUMAN_CUSTOM_LIVER_CODON_TABLE,
            CodonTables::CustomSalivarygland => &HUMAN_CUSTOM_SALIVARYGLAND_CODON_TABLE,
            CodonTables::CustomMuscleSkeletal => &HUMAN_CUSTOM_MUSCLE_SKELETAL_CODON_TABLE,
            CodonTables::CustomTonsil => &HUMAN_CUSTOM_TONSIL_CODON_TABLE,
            CodonTables::CustomSmallintestine => &HUMAN_CUSTOM_SMALLINTESTINE_CODON_TABLE,
            CodonTables::CustomPlacenta => &HUMAN_CUSTOM_PLACENTA_CODON_TABLE,
            CodonTables::CustomAppendices => &HUMAN_CUSTOM_APPENDICES_CODON_TABLE,
            CodonTables::CustomTestis => &HUMAN_CUSTOM_TESTIS_CODON_TABLE,
            CodonTables::CustomRectum => &HUMAN_CUSTOM_RECTUM_CODON_TABLE,
            CodonTables::CustomUrinarybladder => &HUMAN_CUSTOM_URINARYBLADDER_CODON_TABLE,
            CodonTables::CustomProstate => &HUMAN_CUSTOM_PROSTATE_CODON_TABLE,
            CodonTables::CustomEsophagus => &HUMAN_CUSTOM_ESOPHAGUS_CODON_TABLE,
            CodonTables::CustomKidney => &HUMAN_CUSTOM_KIDNEY_CODON_TABLE,
            CodonTables::CustomThyroid => &HUMAN_CUSTOM_THYROID_CODON_TABLE,
            CodonTables::CustomLymphnode => &HUMAN_CUSTOM_LYMPHNODE_CODON_TABLE,
            CodonTables::CustomArtery => &HUMAN_CUSTOM_ARTERY_CODON_TABLE,
            CodonTables::CustomBrain => &HUMAN_CUSTOM_BRAIN_CODON_TABLE,
            CodonTables::CustomNerveTibial => &HUMAN_CUSTOM_NERVE_TIBIAL_CODON_TABLE,
            CodonTables::CustomGallbladder => &HUMAN_CUSTOM_GALLBLADDER_CODON_TABLE,
            CodonTables::CustomUterus => &HUMAN_CUSTOM_UTERUS_CODON_TABLE,
            CodonTables::CustomPituitary => &HUMAN_CUSTOM_PITUITARY_CODON_TABLE,
            CodonTables::CustomColon => &HUMAN_CUSTOM_COLON_CODON_TABLE,
            CodonTables::CustomVagina => &HUMAN_CUSTOM_VAGINA_CODON_TABLE,
            CodonTables::CustomDuodenum => &HUMAN_CUSTOM_DUODENUM_CODON_TABLE,
            CodonTables::CustomFat => &HUMAN_CUSTOM_FAT_CODON_TABLE,
            CodonTables::CustomStomach => &HUMAN_CUSTOM_STOMACH_CODON_TABLE,
            CodonTables::CustomAdrenal => &HUMAN_CUSTOM_ADRENAL_CODON_TABLE,
            CodonTables::CustomFallopiantube => &HUMAN_CUSTOM_FALLOPIANTUBE_CODON_TABLE,
            CodonTables::CustomSmoothmuscle => &HUMAN_CUSTOM_SMOOTHMUSCLE_CODON_TABLE,
            CodonTables::CustomPancreas => &HUMAN_CUSTOM_PANCREAS_CODON_TABLE,
            CodonTables::CustomOvary => &HUMAN_CUSTOM_OVARY_CODON_TABLE,
            CodonTables::EColi => &ECOLI_CODON_TABLE,
            CodonTables::Custom => unsafe { &*CUSTOM_TABLE_PTR },
        }
    }

    /// From a short name return a enum
    /// and the caller must make the the short name is a legal one
    pub fn short2enum(s: &str) -> Self {
        match &s[..] {
            "hc" => Self::Common,
            "as" => Self::AdiposeSubcutaneous,
            "ms" => Self::MuscleSkeletal,
            "at" => Self::ArteryTibial,
            "ac" => Self::ArteryCoronary,
            "haa" => Self::HeartAtrialAppendage,
            "av" => Self::AdiposeVisceral,
            "u" => Self::Uterus,
            "v" => Self::Vagina,
            "b" => Self::BreastMammaryTissue,
            "sns" => Self::SkinNotSunExposed,
            "sse" => Self::SkinSunExposed,
            "msg" => Self::MinorSalivaryGland,
            "bc" => Self::BrainCortex,
            "bcl" => Self::BrainCerebellum,
            "bfc" => Self::BrainFrontalCortex,
            "bce" => Self::BrainCaudate,
            "bna" => Self::BrainNucleusAccumbens,
            "bp" => Self::BrainPutamen,
            "bhp" => Self::BrainHypothalamus,
            "bsc" => Self::BrainSpinalCord,
            "bhi" => Self::BrainHippocampus,
            "bsn" => Self::BrainSubstantiaNigra,
            "bac" => Self::BrainAnteriorCingulateCortex,
            "bam" => Self::BrainAmygdala,
            "bch" => Self::BrainCerebellarHemisphere,
            "ov" => Self::Ovary,
            "ag" => Self::AdrenalGland,
            "th" => Self::Thyroid,
            "lu" => Self::Lung,
            "li" => Self::Liver,
            "kc" => Self::KidneyCortex,
            "sp" => Self::Spleen,
            "pa" => Self::Pancreas,
            "ems" => Self::EsophagusMuscularis,
            "ema" => Self::EsophagusMucosa,
            "egj" => Self::EsophagusGastroesophagealJunction,
            "s" => Self::Stomach,
            "cs" => Self::ColonSigmoid,
            "sit" => Self::SmallIntestineTerminalIleum,
            "ct" => Self::ColonTransverse,
            "pr" => Self::Prostate,
            "ty" => Self::Testis,
            "ne" => Self::NerveTibial,
            "hlv" => Self::HeartLeftVentricle,
            "wb" => Self::WholeBlood,
            "aa" => Self::ArteryAorta,
            "pi" => Self::Pituitary,
            "cec" => Self::CervixEctocervix,
            "ft" => Self::FallopianTube,
            "cen" => Self::CervixEndocervix,
            "clun" => Self::CustomLung,
            "cbre" => Self::CustomBreast,
            "cski" => Self::CustomSkin,
            "cspl" => Self::CustomSpleen,
            "chea" => Self::CustomHeart,
            "cliv" => Self::CustomLiver,
            "csal" => Self::CustomSalivarygland,
            "cmus" => Self::CustomMuscleSkeletal,
            "cton" => Self::CustomTonsil,
            "csma" => Self::CustomSmallintestine,
            "cpla" => Self::CustomPlacenta,
            "capp" => Self::CustomAppendices,
            "ctes" => Self::CustomTestis,
            "crec" => Self::CustomRectum,
            "curi" => Self::CustomUrinarybladder,
            "cpro" => Self::CustomProstate,
            "ceso" => Self::CustomEsophagus,
            "ckid" => Self::CustomKidney,
            "cthy" => Self::CustomThyroid,
            "clym" => Self::CustomLymphnode,
            "cart" => Self::CustomArtery,
            "cbra" => Self::CustomBrain,
            "cner" => Self::CustomNerveTibial,
            "cgal" => Self::CustomGallbladder,
            "cute" => Self::CustomUterus,
            "cpit" => Self::CustomPituitary,
            "ccol" => Self::CustomColon,
            "cvag" => Self::CustomVagina,
            "cduo" => Self::CustomDuodenum,
            "cfat" => Self::CustomFat,
            "csto" => Self::CustomStomach,
            "cadr" => Self::CustomAdrenal,
            "cfal" => Self::CustomFallopiantube,
            "csmo" => Self::CustomSmoothmuscle,
            "cpan" => Self::CustomPancreas,
            "cova" => Self::CustomOvary,
            "ecoli" => Self::EColi,
            _ => {
                // If not a known short name, check if it's a .toml file path.
                if Self::is_toml_path(s) {
                    Self::Custom
                } else {
                    Self::Common
                }
            }
        }
    }
}

/// Take a DNA sequence and return its gc value
/// The caller must make sure the input is an upper case of DNA
fn get_gc_content(cds: &str) -> f64 {
    let cds_size = cds.len();
    let mut gc_nums = 0;
    for c in cds.chars() {
        if c == 'C' || c == 'G' {
            gc_nums += 1;
        }
    }
    (gc_nums as f64) / (cds_size as f64)
}

fn get_u_content(seq: &str) -> f64 {
    let rna = seq.replace("T", "U");
    let rna_size = rna.len();
    let mut u_num = 0;
    for c in rna.chars() {
        if c == 'U' {
            u_num += 1;
        }
    }
    (u_num as f64) / (rna_size as f64)
}

/// Keep track of searching path
#[derive(Clone, Serialize)]
pub struct SearchPath {
    pub seq_id: String,
    pub triplet_seq: Vec<Triplet>,
    pub loss_value: Option<LossInfo>,
    pub codon_table: CodonTables,
    /// Lazily-computed CDS string cache.  Cleared (None) when the path is
    /// created; filled on first `get_cds()` call and carried through Clone.
    #[serde(skip)]
    cached_cds: RefCell<Option<String>>,
}

impl SearchPath {
    pub fn new(t: Vec<Triplet>, seq_id: String, codon_table: CodonTables) -> Self {
        SearchPath {
            seq_id,
            triplet_seq: t,
            loss_value: None,
            codon_table,
            cached_cds: RefCell::new(None),
        }
    }

    /// Give a new triplet and produce a new
    /// mfe not calculated Search Path
    pub fn update_path(&self, new_t: Triplet) -> Self {
        let mut new_seq = self.triplet_seq.clone();
        new_seq.push(new_t);
        SearchPath {
            seq_id: self.seq_id.clone(),
            triplet_seq: new_seq,
            loss_value: None,
            codon_table: self.codon_table.clone(),
            cached_cds: RefCell::new(None),
        }
    }

    /// Take an object of SearchPath and update it's loss value
    pub fn update_loss_value(mut sp: SearchPath, loss: LossInfo) -> Self {
        sp.loss_value = Some(loss);
        sp
    }

    pub fn get_cds(&self) -> String {
        {
            let cached = self.cached_cds.borrow();
            if let Some(ref cds) = *cached {
                return cds.clone();
            }
        }
        let mut outs = Vec::new();
        for t in &self.triplet_seq {
            outs.push(t.to_string());
        }
        let cds = outs.join("");
        *self.cached_cds.borrow_mut() = Some(cds.clone());
        cds
    }

    pub fn get_rna(&self) -> String {
        self.get_cds().replace("T", "U")
    }

    pub fn get_cds_len(&self) -> usize {
        self.triplet_seq.len() * 3
    }

    pub fn get_gc_content(&self) -> f64 {
        let cds = self.get_cds();
        get_gc_content(&cds)
    }

    pub fn get_theoretical_min_max_gc(&self) -> (f64, f64) {
        BeamSearch::_get_min_max_gc(&self.seq_id, self, &self.codon_table)
    }

    /// This method meant to get the secondary structure
    pub fn get_secondary(&self) -> String {
        match self.loss_value {
            Some(ref l) => l.second.clone(),
            None => "".to_string(),
        }
    }

    pub fn get_loss(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.loss,
            None => 0.0,
        }
    }

    pub fn get_mfe(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.mfe,
            None => 0.0,
        }
    }

    // Get the raw cai
    pub fn get_raw_cai(&self) -> f64 {
        call_raw_cai(&self.get_cds()[..], self.codon_table.get_codon_table())
    }

    pub fn get_arithmetic_cai(&self) -> f64 {
        call_raw_arithmetic_cai(&self.get_cds()[..], self.codon_table.get_codon_table())
    }

    pub fn get_palindrome_score(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.palindrome_score,
            None => 0.0,
        }
    }

    pub fn get_palindrome_seqs(&self) -> String {
        let seqs = self
            .loss_value
            .as_ref()
            .unwrap()
            .palindrome_seqs
            .clone()
            .join(";");
        let seqs_num = self.loss_value.as_ref().unwrap().palindrome_seqs.len();
        format!("{}-{}", seqs_num, seqs)
    }

    pub fn get_m2s_score(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.m2s_score,
            None => 0.0,
        }
    }

    pub fn get_raw_m2s_num(&self) -> u64 {
        match self.loss_value {
            Some(ref l) => l.m2s_nums,
            None => 0,
        }
    }

    pub fn get_pll_score(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.pll_score,
            None => 0.0,
        }
    }

    /// Get the raw codon PLL value (original model output, no negation, no weight).
    /// This is the user-facing PLL, analogous to `get_gc_content()` for GC%.
    pub fn get_raw_pll(&self) -> f64 {
        match self.loss_value {
            Some(ref l) => l.raw_pll,
            None => 0.0,
        }
    }

    pub fn to_json(&self) -> Value {
        let cds = self.get_cds();
        let (min_gc, max_gc) = self.get_theoretical_min_max_gc();
        json!({
            "seq_id": self.seq_id.clone(),
            "loss": self.get_loss(),
            "mfe": self.get_mfe(),
            "scaled_cai": call_scaled_cai(&cds, self.codon_table.get_codon_table()),
            "raw_cai": self.get_raw_cai(),
            "arithmetic_cai": self.get_arithmetic_cai(),
            "GC%": self.get_gc_content() * 100.0,
            "optimized_rna_U%": get_u_content(&cds) * 100.0,
            "theoretical_min_GC%": min_gc * 100.0,
            "theoretical_max_GC%": max_gc * 100.0,
            "optimized_cds": cds,
            "optimized_rna": self.get_rna(),
            "optimized_rna_structure": self.get_secondary(),
            "palindrome_score": self.get_palindrome_score(),
            "palindrome_seqs": self.get_palindrome_seqs(),
            "mutate2stop_score": self.get_m2s_score(),
            "mutate2stop_nums": self.get_raw_m2s_num(),
            "pll_score": self.get_pll_score(),
            "raw_pll": self.get_raw_pll(),
        })
    }
}

/// Shared configuration for beam search and SNP mutation search.
#[derive(Clone)]
pub struct SearchConfig {
    pub lambda: f64,
    pub mfe_method: MfeMethod,
    pub weight_gc: f64,
    pub weight_cai: f64,
    pub weight_palindrome: f64,
    pub weight_m2s: f64,
    pub weight_pll: f64,
    pub model_path: Option<String>,
    pub if_minimize_loss: bool,
    pub opt_obj: OptimizeObject,
    /// Which CAI variant is used in the optimization objective.
    pub cai_mode: CaiMode,
    /// Number of initial codons (amino acids) for which MFE and PLL are skipped.
    /// When cur_step < mfe_pll_start_codon, MFE/PLL are treated as zero.
    pub mfe_pll_start_codon: usize,
    /// Number of 5'-end bases (nt) to keep free of stable secondary structure.
    /// 0 = disabled. Must be a multiple of 3. When > 0, the first N bases are
    /// penalized if they form stable RNA structure (low MFE).
    pub weak_head_bases: usize,
}

/// Implement the beam search
#[derive(Clone)]
pub struct BeamSearch {
    /// The input CDS object
    pub cds: CDSSeq,
    /// The maximum searching size
    pub window_size: u32,
    /// The random number when select search-path after exclude the ones in the top `k`
    /// and k is the `window_size`
    pub random_size: u32,
    /// The number of output optimizaed sequences
    pub num_outputs: u32,
    /// The pool to keeping the search-path
    pub search_pool: Vec<SearchPath>,
    /// The unique id of a CDS sequence
    pub seq_id: String,
    /// Avoid sequences
    pub avoid_seqs: Arc<AvoidSeqs>,
    /// The selected codon_table
    pub codon_table: CodonTables,
    /// the progress bar postion for a BeamSearch
    bar_position: u16,
    /// Shared search configuration
    pub config: SearchConfig,
    /// the weight of MFE
    pub weight_mfe: i64,
    /// the thread pool (shared across sequences for efficiency)
    pub thread_pool: Arc<ThreadPool>,
    /// if true, then print out nothing when running the Cosyn
    no_verbose: bool,
}

impl BeamSearch {
    pub fn new(
        cds: CDSSeq,
        win_size: u32,
        random_size: u32,
        num_outputs: u32,
        seq_id: String,
        avoid_seqs: Arc<AvoidSeqs>,
        codon_table: CodonTables,
        bar_position: u16,
        config: SearchConfig,
        thread_pool: Arc<ThreadPool>,
        no_verbose: bool,
    ) -> Self {
        let default_weight_mfe: i64 = 1;
        BeamSearch {
            cds,
            window_size: win_size,
            random_size,
            num_outputs,
            search_pool: Vec::with_capacity(win_size as usize),
            seq_id,
            avoid_seqs,
            codon_table,
            bar_position,
            config,
            weight_mfe: default_weight_mfe,
            thread_pool,
            no_verbose,
        }
    }

    /// Searching the optimal sequences
    pub fn search(&mut self) -> Vec<SearchPath> {
        self._search();
        // After beam search method, the search-path in the pool should be sorted
        // and the size of pool should be window_size + random_size if the input sequence is
        // long enough
        let out_num = if self.window_size < self.num_outputs {
            self.window_size
        } else {
            self.num_outputs
        };
        self.search_pool[..out_num as usize]
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>()
    }

    /// Change the progress bar position
    pub fn set_bar_position(mut self, new_pos: u16) -> Self {
        self.bar_position = new_pos;
        self
    }
}

impl BeamSearch {
    // Take a SearchPath and get its theoretically minimum and maximum GC%
    pub(crate) fn _get_min_max_gc(
        seq_id: &String,
        sp: &SearchPath,
        codon_table: &CodonTables,
    ) -> (f64, f64) {
        let cds = CDSSeq::new(sp.get_cds());
        let original_triplets_num = cds.triplet_num();
        let mut max_search_pool: Vec<SearchPath> = Vec::new();
        let mut min_search_pool: Vec<SearchPath> = Vec::new();
        let mut cur_step = 0;
        while cur_step < original_triplets_num {
            let cur_aa = cds.aa_seq.get(cur_step).unwrap().clone();
            let aa_idx = aa_to_idx(&cur_aa);
            let mut possible_triplets = Vec::new();
            for &idx in CODONS_FOR_AA[aa_idx as usize] {
                possible_triplets.push(CODON_STRINGS[idx as usize].to_string());
            }
            let mut new_min_paths =
                BeamSearch::_next_path(seq_id, codon_table, &min_search_pool, &possible_triplets)
                    .into_iter()
                    .collect::<Vec<_>>();
            let mut new_max_paths =
                BeamSearch::_next_path(seq_id, codon_table, &max_search_pool, &possible_triplets)
                    .into_iter()
                    .collect::<Vec<_>>();
            new_min_paths.sort_by(|s1, s2| {
                let gc1 = s1.get_gc_content();
                let gc2 = s2.get_gc_content();
                gc1.partial_cmp(&gc2).unwrap()
            });
            new_max_paths.sort_by(|s1, s2| {
                let gc1 = s1.get_gc_content();
                let gc2 = s2.get_gc_content();
                gc1.partial_cmp(&gc2).unwrap()
            });
            if !max_search_pool.is_empty() {
                max_search_pool.clear();
            }
            if !min_search_pool.is_empty() {
                min_search_pool.clear();
            }
            max_search_pool.push(new_max_paths.last().unwrap().clone());
            min_search_pool.push(new_min_paths.first().unwrap().clone());
            cur_step += 1;
        }
        (
            min_search_pool.get(0).unwrap().get_gc_content(),
            max_search_pool.get(0).unwrap().get_gc_content(),
        )
    }

    /// The beam searching method
    fn _search(&mut self) {
        let original_triplets_num = self.cds.triplet_num();
        let rnd_threshold = ((original_triplets_num as f64) * RND_WALK_THRESHOLD) as usize;
        let mut cur_step = self.search_pool.len();
        let mut tqdm_bar = tqdm!(
            total = original_triplets_num - cur_step,
            position = self.bar_position
        );
        let display_str_size = if self.seq_id.len() > DISPLAY_STR_LEN {
            DISPLAY_STR_LEN
        } else {
            self.seq_id.len()
        };
        if !self.no_verbose {
            tqdm_bar.set_postfix(format!(
                "Optimizing the {}",
                &self.seq_id[..display_str_size]
            ));
        }
        while cur_step < original_triplets_num {
            let cur_aa = self.cds.aa_seq.get(cur_step).unwrap().clone();
            let aa_idx = aa_to_idx(&cur_aa);
            let mut possible_triplets = Vec::new();
            for &idx in CODONS_FOR_AA[aa_idx as usize] {
                possible_triplets.push(CODON_STRINGS[idx as usize].to_string());
            }

            let mut new_paths: Vec<SearchPath> = self
                ._generate_next_path(&self.search_pool, &possible_triplets)
                .into();
            let mutated_paths;
            if cur_step < rnd_threshold {
                mutated_paths = Vec::new();
            } else {
                mutated_paths = self._random_mutate_path(&new_paths, self.random_size as usize);
            }
            new_paths.extend(mutated_paths);
            let loss_updated_new_paths = self._update_sort_loss(&new_paths, cur_step);
            let filtered_path = self._filter_path(loss_updated_new_paths, true);
            self.search_pool = filtered_path;
            cur_step += 1;
            if !self.no_verbose {
                tqdm_bar.update(1);
            }
        }
    }

    /// A method that will take the current search-path results and the next possible triplets
    /// then produce a bunch of new search-path
    /// And this method will just generate new search-path while not
    /// calculate the mfe
    fn _generate_next_path(
        &self,
        cur_paths: &Vec<SearchPath>,
        next_triplets: &Vec<String>,
    ) -> Vec<SearchPath> {
        BeamSearch::_next_path(&self.seq_id, &self.codon_table, cur_paths, next_triplets)
    }

    fn _next_path(
        seq_id: &String,
        codon_table: &CodonTables,
        cur_paths: &Vec<SearchPath>,
        next_triplets: &Vec<String>,
    ) -> Vec<SearchPath> {
        let mut new_paths = Vec::new();
        if cur_paths.len() == 0 {
            for n in next_triplets {
                //n is the triplet in String type
                new_paths.push(SearchPath::new(
                    vec![BeamSearch::_string2triplet(n)],
                    seq_id.clone(),
                    codon_table.clone(),
                ));
            }
        } else {
            // logically, the last_triplet will not be None
            for prev_path in cur_paths {
                for n in next_triplets {
                    let new_path = prev_path.update_path(BeamSearch::_string2triplet(n));
                    new_paths.push(new_path);
                }
            }
        }
        new_paths
    }

    /// Take a vec of search-path as input and then calculate the loss value
    /// for all the search-path with multiple-threads and return a new and sorted vec of search-path     
    fn _update_sort_loss(&self, paths: &Vec<SearchPath>, cur_step: usize) -> Vec<SearchPath> {
        let n_jobs = paths.len();
        let lambda = self.config.lambda; // the hyperparameter to balance the mfe part and cai part
        let (tx, rx) = channel::<SearchPath>();
        let raw_cds_len = self.cds.raw_seq.len();
        let weight_gc = self.config.weight_gc;
        let weight_cai = self.config.weight_cai;
        let weight_m2s = self.config.weight_m2s;
        let weight_palin = self.config.weight_palindrome;
        let weight_mfe = self.weight_mfe;
        let weight_pll = self.config.weight_pll;
        let skip_mfe_pll = cur_step < self.config.mfe_pll_start_codon;
        for idx in 0..n_jobs {
            let opt_obj = self.config.opt_obj.clone();
            let sender = tx.clone();
            let new_p = paths.get(idx).unwrap().clone();
            let new_cds = new_p.get_cds();
            let new_rna = new_p.get_rna();
            let new_p_len = new_rna.len();
            let codon_table = self.codon_table.get_codon_table();
            let mfe_method = self.config.mfe_method.clone();
            let model_path = self.config.model_path.clone();
            let cai_mode = self.config.cai_mode;
            self.thread_pool.execute(move || match opt_obj {
                OptimizeObject::MfeCai => {
                    let loss = LossFuncs::loss_cai_mfe(
                        &new_p,
                        lambda,
                        codon_table,
                        mfe_method,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        false,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::GcCai => {
                    let loss = LossFuncs::loss_cai_gc(&new_p, codon_table, cai_mode);
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        true,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeGcCai => {
                    let loss = LossFuncs::loss_cai_mfe_gc(
                        lambda,
                        &new_p,
                        codon_table,
                        mfe_method,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        false,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::GcCaiPalin => {
                    let loss = LossFuncs::loss_cai_gc_palin(
                        &new_p,
                        codon_table,
                        weight_cai,
                        weight_gc,
                        weight_palin,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        true,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeCaiPalin => {
                    let loss = LossFuncs::loss_cai_mfe_palin(
                        &new_p,
                        lambda,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_palin,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        false,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeGcCaiPalin => {
                    let loss = LossFuncs::loss_cai_mfe_gc_palin(
                        lambda,
                        &new_p,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_gc,
                        weight_palin,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    if raw_cds_len.eq(&new_p_len) {
                        let m2s_nums = mutate2stop_numbers(&new_cds);
                        let m2s_score = m2s_nums * weight_m2s;
                        loss_updated_path.loss_value.as_mut().unwrap().m2s_score = m2s_score;
                        loss_updated_path.loss_value.as_mut().unwrap().m2s_nums = m2s_nums as u64;
                    }

                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeCaiGcM2sPalin => {
                    let loss = LossFuncs::loss_cai_mfe_gc_m2s_palin(
                        lambda,
                        &new_p,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);

                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::CaiGcM2sPalin => {
                    let loss = LossFuncs::loss_cai_gc_m2s_palin(
                        lambda,
                        &new_p,
                        codon_table,
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    Self::_finalize_step_metrics(
                        &mut loss_updated_path,
                        &new_rna,
                        &new_cds,
                        raw_cds_len,
                        new_p_len,
                        weight_palin,
                        weight_m2s,
                        true,
                        mfe_method,
                        skip_mfe_pll,
                    );
                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
                OptimizeObject::UnifiedObj => {
                    let loss = LossFuncs::loss_unified(
                        lambda,
                        &new_p,
                        codon_table,
                        mfe_method,
                        weight_mfe,
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        model_path.as_deref(),
                        weight_pll,
                        skip_mfe_pll,
                        cai_mode,
                    );
                    let mut loss_updated_path = SearchPath::update_loss_value(new_p, loss);
                    if weight_mfe == 0 {
                        if raw_cds_len.eq(&new_p_len) && !skip_mfe_pll {
                            let (mfe, second) = compute_mfe(&new_rna, mfe_method);
                            loss_updated_path.loss_value.as_mut().unwrap().mfe = mfe;
                            loss_updated_path.loss_value.as_mut().unwrap().second = second;
                        }
                    }

                    sender.send(loss_updated_path).expect(SEND_ERR_MEG);
                }
            });
        }
        let mut loss_updated_paths = rx.into_iter().take(n_jobs).collect::<Vec<_>>();

        // Apply weak-head penalty: penalize stable secondary structure in the
        // 5'-end region. At intermediate steps this acts on the partial
        // sequence; at the final step it acts on the first N bases of the
        // full CDS. This naturally guides the beam away from paths that form
        // stable hairpins near the start codon.
        let head_bases = self.config.weak_head_bases;
        if head_bases > 0 {
            let mfe_method = self.config.mfe_method;
            for sp in &mut loss_updated_paths {
                let rna = sp.get_rna();
                let region = &rna[..rna.len().min(head_bases)];
                let (mfe, _) = compute_mfe(region, mfe_method);
                // stable structure → negative MFE → positive penalty on loss
                sp.loss_value.as_mut().unwrap().loss += (-mfe).max(0.0);
            }
        }

        if self.config.if_minimize_loss {
            loss_updated_paths.sort_by(BeamSearch::_sort_path);
        } else {
            loss_updated_paths.sort_by(BeamSearch::_sort_path_desc);
        }
        loss_updated_paths
    }

    ///
    /// combined sequence is folded with unlimited span, and the normalized
    /// portion of the new fold is added to `loss.loss`.

    /// Apply final-step metrics (palindrome, mutate-to-stop, and optionally MFE)
    /// to a SearchPath whose loss has just been computed.  Only has effect when
    /// the current CDS length equals the original full CDS length.
    fn _finalize_step_metrics(
        sp: &mut SearchPath,
        rna: &str,
        cds: &str,
        raw_cds_len: usize,
        cur_len: usize,
        weight_palin: f64,
        weight_m2s: f64,
        calc_mfe: bool,
        mfe_method: MfeMethod,
        skip_mfe_pll: bool,
    ) {
        if cur_len != raw_cds_len {
            return;
        }
        let loss = sp.loss_value.as_mut().unwrap();

        let (palin_score, palin_seqs) = palindrome_score(rna);
        loss.palindrome_score = palin_score * raw_cds_len as f64 * weight_palin;
        loss.palindrome_seqs = palin_seqs;

        let m2s_nums = mutate2stop_numbers(cds);
        loss.m2s_score = m2s_nums * weight_m2s;
        loss.m2s_nums = m2s_nums as u64;

        if calc_mfe && !skip_mfe_pll {
            let (mfe, second) = compute_mfe(rna, mfe_method);
            loss.mfe = mfe;
            loss.second = second;
        }
    }

    /// The caller must make sure the loss-info is not None
    fn _sort_path(a: &SearchPath, b: &SearchPath) -> Ordering {
        let a_loss = a.loss_value.as_ref().unwrap().loss;
        let b_loss = b.loss_value.as_ref().unwrap().loss;
        if a_loss < b_loss {
            return Ordering::Less;
        } else if a_loss > b_loss {
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
        }
    }

    /// 根据SearchPath的loss值进行降序排序
    ///
    /// 该函数用于比较两个SearchPath实例的loss值，并基于这些值返回它们的排序顺序。
    /// 排序是降序进行的，即loss值较小的实例被认为“更大”，并将排在前面。
    ///
    /// # 参数
    ///
    /// * `a`: &SearchPath - 指向第一个SearchPath实例的引用
    /// * `b`: &SearchPath - 指向第二个SearchPath实例的引用
    ///
    /// # 返回值
    ///
    /// * `Ordering::Greater` - 如果`a`的loss值小于`b`的loss值
    /// * `Ordering::Less` - 如果`a`的loss值大于`b`的loss值
    // * `Ordering::Equal` - 如果`a`和`b`的loss值相等
    fn _sort_path_desc(a: &SearchPath, b: &SearchPath) -> Ordering {
        let a_loss = a.loss_value.as_ref().unwrap().loss;
        let b_loss = b.loss_value.as_ref().unwrap().loss;
        if a_loss < b_loss {
            return Ordering::Greater;
        } else if a_loss > b_loss {
            return Ordering::Less;
        } else {
            return Ordering::Equal;
        }
    }

    /// Remove SearchPath with unwanted seqs and sort and select top k
    /// if `regenerate` true, this function will try to mutate the SearchPath
    /// if the Vec to store the SearchPath is empty after filtering
    /// if `regenerate` false, this function will ignore the mutation part
    fn _filter_path(&self, sorted_paths: Vec<SearchPath>, regenerate: bool) -> Vec<SearchPath> {
        // remove SearchPath with avoid sequences
        let sorted_paths_cloned = sorted_paths.clone();
        let unwanted_seqs = self.avoid_seqs.get_unwanted_seqs();
        let mut filtered_paths = sorted_paths
            .iter()
            .filter(|s| {
                let in_seq = &s.get_cds();
                self.avoid_seqs.filter_cds(in_seq) && self.avoid_seqs.filter_homopolymers(in_seq)
            })
            .map(|x| x.clone())
            .collect::<Vec<_>>();

        if regenerate {
            let mut sp_with_unwanted = sorted_paths_cloned.clone();
            'outer: loop {
                if filtered_paths.len() == 0 {
                    // unwanted motif start index in the cds
                    let mut pos_mutated_paths = Vec::new();
                    for sp in sp_with_unwanted {
                        let template_cds = sp.get_cds();
                        for motif in &unwanted_seqs {
                            for start_idx in BeamSearch::_find_unwanted_cds_index(&sp, motif) {
                                pos_mutated_paths
                                    .extend(self._mutate_pos(&template_cds, start_idx));
                            }
                        }
                    }
                    if self.config.if_minimize_loss {
                        pos_mutated_paths.sort_by(BeamSearch::_sort_path);
                    } else {
                        pos_mutated_paths.sort_by(BeamSearch::_sort_path_desc);
                    }
                    filtered_paths = pos_mutated_paths
                        .clone()
                        .into_iter()
                        .filter(|s| {
                            let in_seq = &s.get_cds();
                            self.avoid_seqs.filter_cds(in_seq)
                                && self.avoid_seqs.filter_homopolymers(in_seq)
                        })
                        .collect::<Vec<_>>();
                    sp_with_unwanted = pos_mutated_paths;
                    continue 'outer;
                } else {
                    break 'outer;
                }
            }
        }

        let sorted_paths = filtered_paths;

        let top_k_nums = self.window_size;
        let random_nums = self.random_size;
        let input_path_size = sorted_paths.len();
        let cap = if ((top_k_nums + random_nums) as usize) < input_path_size {
            (top_k_nums + random_nums) as usize
        } else {
            input_path_size
        };
        let mut filter_outs = Vec::with_capacity(cap);
        // how many smallest top k to select for next step
        let pop_nums = if input_path_size < (top_k_nums as usize) {
            input_path_size
        } else {
            top_k_nums as usize
        };
        // First, select the smallest ones for the next step
        for sp in &sorted_paths[0..pop_nums] {
            filter_outs.push(sp.clone());
        }
        filter_outs
    }

    /// Find the start index in the CDS that starts the unwanted motif sequence
    /// With the motif the function will fist locat the `wrong amino acids indexes`
    /// and then return the `first codon index` in that `wrong amino acids indexes`
    fn _find_unwanted_cds_index(sp: &SearchPath, motif: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let cds = sp.get_cds();
        let motif_size = motif.len();
        for i in 0..cds.len() {
            let end;
            if i + motif_size <= cds.len() {
                end = i + motif_size;
            } else {
                end = cds.len();
            }
            if (&cds[i..end]).eq(motif) {
                for j in i..end {
                    let aa_idx = j / 3;
                    let first_codon_idx = aa_idx * 3;
                    if !out.contains(&first_codon_idx) {
                        out.push(first_codon_idx);
                    }
                }
            }
        }
        out
    }

    /// Give a CDS an index in the CDS, mutate the triplet to avoid unwanted motifs
    /// The caller must make sure the `start_idx` is within the boundary of CDS
    /// and this function will return a non-redundant Vec<SearchPath>  
    fn _mutate_pos(&self, cds: &str, start_idx: usize) -> Vec<SearchPath> {
        let triplet = cds[start_idx..start_idx + 3].to_string();
        let aa = tri2aa(&triplet[..]);
        let aa_pos = start_idx / 3;
        let aa_idx = aa_to_idx(&aa);
        let rest_triplets: Vec<String> = CODONS_FOR_AA[aa_idx as usize]
            .iter()
            .map(|&idx| CODON_STRINGS[idx as usize].to_string())
            .filter(|t| t.ne(&triplet))
            .collect();
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let cds_chars = cds.chars().collect::<Vec<_>>();
        for new_tri in rest_triplets {
            let mut new_cds = Vec::new();
            for (raw_aa_idx, raw_tri) in cds_chars.chunks(3).enumerate() {
                let raw_tri = String::from_iter(raw_tri);
                if raw_aa_idx.ne(&aa_pos) {
                    new_cds.push(BeamSearch::_string2triplet(&raw_tri));
                } else {
                    new_cds.push(BeamSearch::_string2triplet(&new_tri));
                }
            }
            let new_sp = SearchPath::new(new_cds, self.seq_id.clone(), self.codon_table.clone());
            let new_cds_seq = new_sp.get_cds();
            if !seen.contains(&new_cds_seq) {
                seen.insert(new_cds_seq);
                out.push(new_sp);
            }
        }
        self._update_sort_loss(&out, usize::MAX)
    }

    /// Take the new generated SearchPath and try to generate randomly mutated SearchPath
    /// from it.
    fn _random_mutate_path(&self, sp: &[SearchPath], rnd_num: usize) -> Vec<SearchPath> {
        let mut filtered_mutated = Vec::new();
        let mut rng = rand::thread_rng();
        for s in sp {
            let mutated_sp = self._change_codon(s);
            if BeamSearch::_sp_not_contains(&mutated_sp, sp) {
                filtered_mutated.push(mutated_sp);
            }
        }
        self._filter_path(filtered_mutated, false)
            .into_iter()
            .choose_multiple(&mut rng, rnd_num)
    }

    // Try to imporve the GC content of the input SearchPath if possible
    /// or at least change the codon to a new codon with the same GC content
    fn _change_codon(&self, sp: &SearchPath) -> SearchPath {
        let mut rng = rand::thread_rng();
        let mut new_tri_seq = Vec::new();
        let cds = sp.get_cds().chars().collect::<Vec<_>>();
        for tri in cds.chunks(3) {
            let codon = String::from_iter(tri); // current codon
            let amino_acid = tri2aa(&codon);
            let aa_idx = aa_to_idx(&amino_acid);
            let mut new_codons: Vec<String> = CODONS_FOR_AA[aa_idx as usize]
                .iter()
                .map(|&idx| CODON_STRINGS[idx as usize].to_string())
                .filter(|t| t.ne(&codon))
                .collect();
            new_codons.shuffle(&mut rng);
            let new_codon = match new_codons.first() {
                Some(nc) => nc.clone(),
                None => codon.clone(),
            };
            if new_codon.eq(&codon) {
                new_tri_seq.push(codon);
            } else {
                if BeamSearch::_cal_codon_gc(&new_codon) < BeamSearch::_cal_codon_gc(&codon) {
                    new_tri_seq.push(codon);
                } else {
                    // GC(new_codon) >= GC(codon)
                    let rnd_num = rng.gen_range(0..10); // a random number
                    let rnd_threshold = rng.gen_range(0..10);
                    if rnd_num > rnd_threshold {
                        // randomly keep the new_codon or just use the previous codon
                        new_tri_seq.push(new_codon);
                    } else {
                        new_tri_seq.push(codon);
                    }
                }
            }
        }
        SearchPath::new(
            CDSSeq::new(new_tri_seq.join("")).triplet_seq.into(),
            self.seq_id.clone(),
            self.codon_table.clone(),
        )
    }

    /// Given a Codon string calculate its gc content
    /// smaller is better so times -1.0
    fn _cal_codon_gc(codon: &str) -> f64 {
        let mut gc_num = 0;
        for c in codon.chars() {
            if c == 'G' || c == 'C' {
                gc_num += 1;
            }
        }
        (gc_num as f64) / (codon.len() as f64 * -1.0)
    }

    /// The new `sp` from mutation should not be contained in the `prev_sp`
    /// return `true` means it's good
    fn _sp_not_contains(sp: &SearchPath, prev_sp: &[SearchPath]) -> bool {
        let check_cds = sp.get_cds();
        for p in prev_sp {
            if p.get_cds().eq(&check_cds) {
                return false;
            }
        }
        return true;
    }

    /// Take a String and make a Triplet
    /// and the caller shall make sure the
    /// input String is a triplet but in String
    /// type
    fn _string2triplet(s: &str) -> Triplet {
        let char_s: Vec<char> = s.chars().collect();
        Triplet::new(char_s[0], char_s[1], char_s[2])
    }
}

/// A struct to store the partial optimizing method.
/// Given a template of CDS sequence this struct aims to
/// only change some specific amino acids in this sequence
/// in a way that optimize the SNP mutated CDS as could as possible
/// while not change other bases in the template sequence
#[derive(Clone)]
pub(crate) struct SnpMutation {
    /// sequence name
    seq_id: String,
    /// The original positions in the AA sequence that needed to be mutated
    /// It will be extracted from User's input like `A121R` and the pos is the 121.
    /// please note the positions are start from 1 not 0. [1, seq_size]
    mutation_aa_pos: Vec<usize>,
    /// The target amino acids that want to mutate to be, and it's order must be consistent with that of `aa_sites`
    target_aa: Vec<String>,
    /// The template CDS sequece
    template_cds_seq: CDSSeq,
    /// The selected codon table
    codon_table: CodonTables,
    /// A hash-table to store the `target mutation aa index` -> `possible codon of that aa`
    /// And just like the `mutation_aa_pos` the position are user provided and 1 started
    aa_pos_codons: HashMap<usize, Vec<String>>,
    /// The thread pool object
    thread_pool: ThreadPool,
    /// Avoid sequences
    avoid_seqs: Arc<AvoidSeqs>,
    /// Shared search configuration
    pub config: SearchConfig,
}

impl SnpMutation {
    /// The caller should do some basic check before calling this new method
    pub(crate) fn new(
        seq_id: String,
        mutation_aa_pos: Vec<usize>,
        target_aa: Vec<String>,
        template_cds_seq: CDSSeq,
        codon_table: CodonTables,
        thread_pool: ThreadPool,
        avoid_seqs: Arc<AvoidSeqs>,
        config: SearchConfig,
    ) -> Self {
        SnpMutation::init_aa_index_codons(SnpMutation {
            seq_id,
            mutation_aa_pos,
            target_aa,
            template_cds_seq,
            codon_table,
            aa_pos_codons: HashMap::new(),
            thread_pool,
            avoid_seqs,
            config,
        })
    }

    #[allow(dead_code)]
    pub fn set_loss_type(&mut self, loss_type: bool) {
        self.config.if_minimize_loss = loss_type;
    }

    /// Init the table of `AA index` -> `possible codons`
    /// The caller must make sure the `aa_sites` and `target_aa` are consistent
    fn init_aa_index_codons(mut self) -> Self {
        for i in 0..self.mutation_aa_pos.len() {
            let cur_aa = &self.target_aa[i];
            let cur_aa_pos = *(&self.mutation_aa_pos[i]); // the mutation aa position from user input
            let aa_idx = aa_to_idx(&cur_aa);
            let cur_codons: Vec<String> = CODONS_FOR_AA[aa_idx as usize]
                .iter()
                .map(|&idx| CODON_STRINGS[idx as usize].to_string())
                .collect();
            self.aa_pos_codons.insert(cur_aa_pos, cur_codons);
        }
        self
    }

    /// Search the best SNP SearchPath
    pub(crate) fn search_best(&self) -> SearchPath {
        let weight_gc = self.config.weight_gc;
        let weight_cai = self.config.weight_cai;
        let weight_palin = self.config.weight_palindrome;
        let weight_m2s = self.config.weight_m2s;
        let weight_pll = self.config.weight_pll;
        let mut search_path_iter = SnpMutIter::new(&self);
        let n_jobs = search_path_iter.search_size;
        let (tx, rx) = channel::<SearchPath>();

        let mut all_nxt_sp = Vec::new();
        while let Some(sp) = search_path_iter.next() {
            all_nxt_sp.push(sp);
        }

        while let Some(sp) = search_path_iter.next() {
            let lambda = self.config.lambda;
            let opt_obj = self.config.opt_obj.clone();
            let codon_table = self.codon_table.get_codon_table();
            let sender = tx.clone();
            let mfe_method = self.config.mfe_method.clone();
            let model_path = self.config.model_path.clone();
            let cai_mode = self.config.cai_mode;
            self.thread_pool.execute(move || match opt_obj {
                OptimizeObject::MfeCai => {
                    let loss = LossFuncs::loss_cai_mfe(
                        &sp,
                        lambda,
                        codon_table,
                        mfe_method,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeCaiPalin => {
                    let loss = LossFuncs::loss_cai_mfe_palin(
                        &sp,
                        lambda,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_palin,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::GcCai => {
                    let loss = LossFuncs::loss_cai_gc(&sp, codon_table, cai_mode);
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::GcCaiPalin => {
                    let loss = LossFuncs::loss_cai_gc_palin(
                        &sp,
                        codon_table,
                        weight_cai,
                        weight_gc,
                        weight_palin,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeGcCai => {
                    let loss = LossFuncs::loss_cai_mfe_gc(
                        lambda,
                        &sp,
                        codon_table,
                        mfe_method,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeGcCaiPalin => {
                    let loss = LossFuncs::loss_cai_mfe_gc_palin(
                        lambda,
                        &sp,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_gc,
                        weight_palin,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::MfeCaiGcM2sPalin => {
                    let loss = LossFuncs::loss_cai_mfe_gc_m2s_palin(
                        lambda,
                        &sp,
                        codon_table,
                        mfe_method,
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::CaiGcM2sPalin => {
                    let loss = LossFuncs::loss_cai_gc_m2s_palin(
                        lambda,
                        &sp,
                        codon_table,
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
                OptimizeObject::UnifiedObj => {
                    let loss = LossFuncs::loss_unified(
                        lambda,
                        &sp,
                        codon_table,
                        mfe_method,
                        1, // default weight_mfe for SNP
                        weight_cai,
                        weight_gc,
                        weight_m2s,
                        weight_palin,
                        model_path.as_deref(),
                        weight_pll,
                        false,
                        cai_mode,
                    );
                    let loss_updated_sp = SearchPath::update_loss_value(sp, loss);
                    sender.send(loss_updated_sp).expect(SEND_ERR_MEG);
                }
            })
        }
        let mut loss_updated_paths = rx.into_iter().take(n_jobs).collect::<Vec<_>>();
        if self.config.if_minimize_loss {
            loss_updated_paths.sort_by(BeamSearch::_sort_path);
        } else {
            loss_updated_paths.sort_by(BeamSearch::_sort_path_desc);
        }

        let filtered_sp = loss_updated_paths
            .into_iter()
            .filter(|s| {
                let in_seq = &s.get_cds();
                self.avoid_seqs.filter_cds(in_seq) && self.avoid_seqs.filter_homopolymers(in_seq)
            })
            .collect::<Vec<_>>();

        filtered_sp.first().unwrap().clone()
    }
}

/// A HashMap to store the mutation positions from the user input which started from 1
/// and the values are the accessible index for that amino acid(from the input position)
/// to access the possible codons and it's start from 0.
/// For example, say the user provide `N4A` which means the user want to change the `N` at the `4` position
/// to `A` and for `A` it has four possible codons vec!["GCT", "GCC", "GCA", "GCG"]
/// so the value in this map when the key equals to 4 is within [0, 3]
/// like HashMap(4, 0) or HashMap(4, 3) are good
type SnpPosIdx = HashMap<usize, usize>;

#[derive(Clone)]
struct PosIdx {
    /// user input aa postion
    pos: usize,
    /// current codon index for that aa
    idx: usize,
}

struct SnpMutIter<'a> {
    snp_mut: &'a SnpMutation,
    pub(crate) search_size: usize,
    /// The maximum index for each position in a aa seq.
    /// For example, if the target aa is `A` they will be four codons to select
    /// then the maximum index here is `3`
    /// The current codon index for the amino acid at the input position.
    current_pos_idx: SnpPosIdx,
    idx_combinations: MultiProduct<IntoIter<PosIdx>>,
}

impl<'a> SnpMutIter<'a> {
    fn new(sm: &'a SnpMutation) -> Self {
        let mut combination_size = 1;
        let mut all_pos_idx = Vec::new();
        for (pos, codons) in &sm.aa_pos_codons {
            combination_size *= codons.len();
            all_pos_idx.push(
                (0..codons.len())
                    .map(|codon_idx| PosIdx {
                        pos: *pos,
                        idx: codon_idx,
                    })
                    .collect::<Vec<PosIdx>>(),
            );
        }
        let mut current_pos_idx = HashMap::new();
        for (pos, _codons) in &sm.aa_pos_codons {
            current_pos_idx.insert(*pos, 0);
        }
        let combine_iter = all_pos_idx.into_iter().multi_cartesian_product();
        SnpMutIter {
            snp_mut: sm,
            search_size: combination_size,
            idx_combinations: combine_iter,
            current_pos_idx,
        }
    }

    /// Get the new cds sequence when the index have been updated
    fn get_new_search_path(&self) -> SearchPath {
        let seq_id = self.snp_mut.seq_id.clone();
        let codon_table = self.snp_mut.codon_table.clone();
        let mut outs = Vec::new();
        let template_cds = self.snp_mut.template_cds_seq.get_cds();
        let mutation_aa_pos = self.snp_mut.mutation_aa_pos.clone();
        let template_cds_size = template_cds.len();
        for cds_idx in 0..template_cds_size {
            if (cds_idx % 3) == 0 {
                if !mutation_aa_pos.contains(&((cds_idx / 3) + 1)) {
                    // cds_idx / 3 get the first aa index in the cds string
                    // and this time it's not a mutation site
                    let unmutated_sub_cds = template_cds[cds_idx..cds_idx + 3].to_string();
                    outs.push(Triplet::from_string(unmutated_sub_cds));
                } else {
                    // the cur_aa_idx is a mutation site
                    let cur_aa_idx = cds_idx / 3;
                    let cur_aa_pos = cur_aa_idx + 1;
                    let cur_codon_idx = *self.current_pos_idx.get(&cur_aa_pos).unwrap();
                    let cur_codon =
                        self.snp_mut.aa_pos_codons.get(&cur_aa_pos).unwrap()[cur_codon_idx].clone();
                    outs.push(Triplet::from_string(cur_codon));
                }
            }
        }
        SearchPath::new(outs, seq_id, codon_table)
    }
}

impl<'a> Iterator for SnpMutIter<'a> {
    type Item = SearchPath;
    fn next(&mut self) -> Option<Self::Item> {
        match self.idx_combinations.next() {
            Some(new_current_pi) => {
                for pi in new_current_pi {
                    let pos = pi.pos;
                    let new_cur_idx = pi.idx;
                    if let Some(codon_idx) = self.current_pos_idx.get_mut(&pos) {
                        *codon_idx = new_cur_idx;
                    }
                }
                Some(self.get_new_search_path())
            }
            None => None,
        }
    }
}

//! Core DP state types.

pub type Score = i32;

/// Sentinel value representing negative infinity for DP scores.
pub const SCORE_MIN: Score = Score::MIN;

/// Traceback label indicating how a state was derived.
///
/// Variant values are kept identical to the C++ `Manner` enum for easier
/// cross-reference with the original implementation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(i32)]
pub enum Manner {
    #[default]
    None = 0,
    H = 1,
    Hairpin = 2,
    Single = 3,
    Helix = 4,
    Multi = 5,
    MultiEqMultiPlusU = 6,
    PEqMulti = 7,
    M2EqMPlusP = 8,
    MEqM2 = 9,
    MEqMPlusU = 10,
    MEqP = 11,
    CEqCPlusU = 12,
    CEqCPlusP = 13,
}

/// Auxiliary traceback information.
///
/// C++ stores this in a `union`. We use a Rust enum for type safety.
#[derive(Clone, Copy, Debug)]
pub enum TraceInfo {
    Split { split: i32 },
    Paddings { l1: i8, l2: i32 },
}

impl Default for TraceInfo {
    fn default() -> Self {
        TraceInfo::Split { split: -1 }
    }
}

/// A single dynamic-programming state.
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub score: Score,
    pub manner: Manner,
    pub trace: TraceInfo,
}

impl Default for State {
    fn default() -> Self {
        State {
            score: SCORE_MIN,
            manner: Manner::None,
            trace: TraceInfo::default(),
        }
    }
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, score: Score, manner: Manner) {
        self.score = score;
        self.manner = manner;
    }

    pub fn set_split(&mut self, score: Score, manner: Manner, split: i32) {
        self.score = score;
        self.manner = manner;
        self.trace = TraceInfo::Split { split };
    }

    pub fn set_paddings(&mut self, score: Score, manner: Manner, l1: i8, l2: i32) {
        self.score = score;
        self.manner = manner;
        self.trace = TraceInfo::Paddings { l1, l2 };
    }

    /// Update this state if `newscore` is better.
    pub fn update_if_better(&mut self, newscore: Score, manner: Manner) {
        if self.score < newscore {
            self.set(newscore, manner);
        }
    }

    /// Update if better, also storing a split index.
    pub fn update_if_better_split(&mut self, newscore: Score, manner: Manner, split: i32) {
        if self.score < newscore || matches!(self.manner, Manner::None) {
            self.set_split(newscore, manner, split);
        }
    }

    /// Update if better, also storing loop paddings.
    pub fn update_if_better_paddings(&mut self, newscore: Score, manner: Manner, l1: i8, l2: i32) {
        if self.score < newscore || matches!(self.manner, Manner::None) {
            self.set_paddings(newscore, manner, l1, l2);
        }
    }
}

/// Result returned by `LinearFold::parse()`.
#[derive(Debug)]
pub struct DecoderResult {
    pub structure: String,
    pub score: Score,
}

/// A beam is a sparse set of states keyed by their left endpoint.
pub type Beam = rustc_hash::FxHashMap<i32, State>;

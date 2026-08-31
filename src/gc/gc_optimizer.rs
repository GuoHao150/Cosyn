use crate::beam_search::{CodonTables, SearchPath};
use crate::cds::CDSSeq;
use serde_json::{json, Value};

/// The GC optimizer this will try to maximize the GC content
pub(crate) struct GcOptimizer<'a> {
    search_path: &'a SearchPath,
}

impl<'a> GcOptimizer<'a> {
    pub(crate) fn new(sp: &'a SearchPath) -> Self {
        GcOptimizer { search_path: sp }
    }

    /// Calculate the loss of GC which is negative of the gc content
    /// the smaller it is the better the results will be negative
    pub(crate) fn gc_abs_loss(&self) -> f64 {
        (-1.0) * self.get_gc_content()
    }

    pub(crate) fn get_gc_content(&self) -> f64 {
        self.search_path.get_gc_content()
    }
}

pub(crate) struct GcCalculator<'a> {
    seq_id: &'a String,
    _search_path: SearchPath,
}

impl<'a> GcCalculator<'a> {
    pub(crate) fn new(seq_id: &'a String, seq: &'a String) -> Self {
        let _dummy_table = CodonTables::Common;
        let sp = SearchPath::new(
            CDSSeq::new(seq).triplet_seq.into(),
            seq_id.to_string(),
            _dummy_table,
        );
        GcCalculator {
            seq_id,
            _search_path: sp,
        }
    }

    fn get_gc(&self) -> f64 {
        self._search_path.get_gc_content() * 100.0
    }

    fn get_min_max_gc(&self) -> (f64, f64) {
        let (min, max) = self._search_path.get_theoretical_min_max_gc();
        (min * 100.0, max * 100.0)
    }

    pub(crate) fn to_json(&self) -> Value {
        let (min_gc, max_gc) = self.get_min_max_gc();
        json!({
            "seq_id": self.seq_id.clone(),
            "GC%": self.get_gc(),
            "theoretical_min_GC%": min_gc,
            "theoretical_max_GC%": max_gc,
        })
    }
}

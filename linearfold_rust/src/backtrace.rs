//! Traceback from the final C state to a dot-bracket structure.

use crate::parser::LinearFold;
use crate::state::{Manner, State, TraceInfo};

impl LinearFold {
    pub(crate) fn get_parentheses(&self, _seq: &str) -> String {
        let mut result: Vec<char> = vec!['.'; self.seq_length];
        let mut stk: Vec<(usize, usize, State)> = Vec::new();
        let final_state = self.best_c[self.seq_length - 1];
        stk.push((0, self.seq_length - 1, final_state));

        while let Some((i, j, state)) = stk.pop() {
            match state.manner {
                Manner::H => {}
                Manner::Hairpin => {
                    result[i] = '(';
                    result[j] = ')';
                }
                Manner::Single => {
                    if let TraceInfo::Paddings { l1, l2 } = state.trace {
                        let p = i + l1 as usize;
                        let q = j - l2 as usize;
                        result[i] = '(';
                        result[j] = ')';
                        if let Some(&inner_state) = self.best_p[q].get(&(p as i32)) {
                            stk.push((p, q, inner_state));
                        }
                    }
                }
                Manner::Helix => {
                    result[i] = '(';
                    result[j] = ')';
                    if let Some(&inner_state) = self.best_p[j - 1].get(&((i + 1) as i32)) {
                        stk.push((i + 1, j - 1, inner_state));
                    }
                }
                Manner::Multi | Manner::MultiEqMultiPlusU => {
                    if let TraceInfo::Paddings { l1, l2 } = state.trace {
                        let p = i + l1 as usize;
                        let q = j - l2 as usize;
                        if let Some(&inner_state) = self.best_m2[q].get(&(p as i32)) {
                            stk.push((p, q, inner_state));
                        }
                    }
                }
                Manner::PEqMulti => {
                    result[i] = '(';
                    result[j] = ')';
                    if let Some(&inner_state) = self.best_multi[j].get(&(i as i32)) {
                        stk.push((i, j, inner_state));
                    }
                }
                Manner::M2EqMPlusP => {
                    if let TraceInfo::Split { split } = state.trace {
                        let k = split as usize;
                        if let Some(&m_state) = self.best_m[k].get(&(i as i32)) {
                            stk.push((i, k, m_state));
                        }
                        if let Some(&p_state) = self.best_p[j].get(&((k + 1) as i32)) {
                            stk.push((k + 1, j, p_state));
                        }
                    }
                }
                Manner::MEqM2 => {
                    if let Some(&inner_state) = self.best_m2[j].get(&(i as i32)) {
                        stk.push((i, j, inner_state));
                    }
                }
                Manner::MEqMPlusU => {
                    if j > 0 {
                        if let Some(&inner_state) = self.best_m[j - 1].get(&(i as i32)) {
                            stk.push((i, j - 1, inner_state));
                        }
                    }
                }
                Manner::MEqP => {
                    if let Some(&inner_state) = self.best_p[j].get(&(i as i32)) {
                        stk.push((i, j, inner_state));
                    }
                }
                Manner::CEqCPlusU => {
                    if j > 0 {
                        stk.push((0, j - 1, self.best_c[j - 1]));
                    }
                }
                Manner::CEqCPlusP => {
                    if let TraceInfo::Split { split } = state.trace {
                        let k = split;
                        if k != -1 {
                            let k = k as usize;
                            stk.push((0, k, self.best_c[k]));
                            if let Some(&p_state) = self.best_p[j].get(&((k + 1) as i32)) {
                                stk.push((k + 1, j, p_state));
                            }
                        } else if let Some(&p_state) = self.best_p[j].get(&(i as i32)) {
                            stk.push((i, j, p_state));
                        }
                    }
                }
                Manner::None => {}
            }
        }

        result.into_iter().collect()
    }
}

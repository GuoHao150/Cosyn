#!/usr/bin/env python3
"""Generate src/cai/table.rs from the HIVE Tissue Codon Usage TSV.

Source paper:
    A new and updated resource for codon usage tables

The script expects the source file:
    ./o537-Tissue_Codon_2023_3_8.tsv

and writes:
    ./src/cai/table.rs

The output uses a compact 6-bit codon encoding (A=0, C=1, G=2, T=3) and
stores each table as a `CodonTable { freqs: [f64; 64], max_freq: [f64; 21] }`.
"""
import json
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq

# 6-bit codon encoding: A=0, C=1, G=2, T=3
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# Stable amino-acid ordering used by the Rust code.  Index 20 is the stop codon.
AA_ORDER = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def codon_to_idx(codon: str) -> int:
    """Encode a DNA codon as a 6-bit integer."""
    b0 = BASE_MAP[codon[0]]
    b1 = BASE_MAP[codon[1]]
    b2 = BASE_MAP[codon[2]]
    return (b0 << 4) | (b1 << 2) | b2


def idx_to_codon(idx: int) -> str:
    """Decode a 6-bit codon index back to a 3-letter DNA string."""
    bits_to_base = {0: "A", 1: "C", 2: "G", 3: "T"}
    b0 = (idx >> 4) & 0b11
    b1 = (idx >> 2) & 0b11
    b2 = idx & 0b11
    return bits_to_base[b0] + bits_to_base[b1] + bits_to_base[b2]


def build_freq_array(table: dict) -> list:
    """Flatten {aa: {codon: freq}} into a [f64; 64] array."""
    freqs = [0.0] * 64
    for aa, codons in table.items():
        for codon, freq in codons.items():
            freqs[codon_to_idx(codon)] = freq
    return freqs


def build_max_freq(table: dict) -> list:
    """Compute the maximum frequency for each amino acid."""
    return [max(table.get(aa, {"": 0.0}).values()) if aa in table else 0.0 for aa in AA_ORDER]


def main():
    root = Path(__file__).resolve().parents[2]
    tsv_path = root / "o537-Tissue_Codon_2023_3_8.tsv"
    out_path = root / "src" / "cai" / "table.rs"

    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    new_index = [i.replace(" - ", "_").replace(" ", "_") for i in df.index]
    df.index = new_index
    df = df[list(df.columns)[:-1]]
    df_t = df.T.copy()
    df_t["aa"] = [str(Seq(x).translate()) for x in df_t.index]

    grouped = df_t.groupby("aa")
    demo_f = grouped.get_group("F")
    target_tissues = list(demo_f.columns)[:-2]

    scaled_dfs = {}
    for aa, d in grouped:
        d = d[target_tissues]
        scaled_d = d / d.sum(axis=0)
        scaled_dfs[aa] = scaled_d.to_dict()

    cache_json = root / "o537-Tissue_scaled_Codon_2023_3_8.json"
    with open(cache_json, "w") as fw:
        json.dump(scaled_dfs, fw, indent=4, sort_keys=True)

    # Shared genetic-code constants derived from the first table.
    genetic = scaled_dfs
    aa_of = [255] * 64
    codons_for_aa = {aa: [] for aa in AA_ORDER}
    for aa in AA_ORDER:
        if aa not in genetic:
            continue
        for codon in genetic[aa]:
            idx = codon_to_idx(codon)
            aa_of[idx] = AA_TO_IDX[aa]
            codons_for_aa[aa].append(idx)

    lines = []
    lines.append("//! Auto-generated compact codon frequency tables.")
    lines.append("//! Generated from o537-Tissue_Codon_2023_3_8.tsv by src/cai/update_codon_table.py")
    lines.append("")
    lines.append("/// Compact codon frequency table.")
    lines.append("///")
    lines.append("/// Codons are encoded as 6-bit integers: A=0, C=1, G=2, T=3,")
    lines.append("/// so index = (base0 << 4) | (base1 << 2) | base2, range 0..64.")
    lines.append("pub struct CodonTable {")
    lines.append("    /// Frequency for each of the 64 codons.")
    lines.append("    pub freqs: [f64; 64],")
    lines.append("    /// Precomputed maximum frequency for each amino acid (see `AA_CHARS`).")
    lines.append("    pub max_freq: [f64; 21],")
    lines.append("}")
    lines.append("")
    lines.append("impl CodonTable {")
    lines.append("    /// Frequency of a codon given its 6-bit index.")
    lines.append("    #[inline]")
    lines.append("    pub fn freq_by_idx(&self, idx: u8) -> f64 {")
    lines.append("        self.freqs[idx as usize]")
    lines.append("    }")
    lines.append("")
    lines.append("    /// Frequency of a codon given as a 3-byte DNA string.")
    lines.append("    #[inline]")
    lines.append("    #[allow(dead_code)]")
    lines.append("    pub fn freq(&self, codon: &[u8]) -> f64 {")
    lines.append("        self.freqs[codon_to_idx(codon) as usize]")
    lines.append("    }")
    lines.append("")
    lines.append("    /// CAI ratio (freq / max_freq_for_aa) for a codon string.")
    lines.append("    #[inline]")
    lines.append("    pub fn cai_ratio(&self, codon: &[u8]) -> f64 {")
    lines.append("        let idx = codon_to_idx(codon);")
    lines.append("        self.freqs[idx as usize] / self.max_freq[AA_OF_CODON[idx as usize] as usize]")
    lines.append("    }")
    lines.append("")
    lines.append("    /// Precomputed maximum frequency for an amino-acid index.")
    lines.append("    #[inline]")
    lines.append("    #[allow(dead_code)]")
    lines.append("    pub fn max_freq_for_aa_idx(&self, aa_idx: u8) -> f64 {")
    lines.append("        self.max_freq[aa_idx as usize]")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("/// Encode a DNA base as 2 bits.")
    lines.append("#[inline]")
    lines.append("const fn base_to_bits(base: u8) -> u8 {")
    lines.append("    match base {")
    lines.append("        b'A' | b'a' => 0,")
    lines.append("        b'C' | b'c' => 1,")
    lines.append("        b'G' | b'g' => 2,")
    lines.append("        b'T' | b't' | b'U' | b'u' => 3,")
    lines.append("        _ => panic!(\"invalid DNA base\"),")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("/// Encode a 3-base DNA codon as a 6-bit index.")
    lines.append("#[inline]")
    lines.append("pub const fn codon_to_idx(codon: &[u8]) -> u8 {")
    lines.append("    (base_to_bits(codon[0]) << 4) | (base_to_bits(codon[1]) << 2) | base_to_bits(codon[2])")
    lines.append("}")
    lines.append("")
    lines.append("/// Decode a 6-bit index back to a 3-base DNA codon.")
    lines.append("#[inline]")
    lines.append("#[allow(dead_code)]")
    lines.append("pub const fn idx_to_codon(idx: u8) -> [u8; 3] {")
    lines.append("    const BITS_TO_BASE: [u8; 4] = [b'A', b'C', b'G', b'T'];")
    lines.append("    [")
    lines.append("        BITS_TO_BASE[((idx >> 4) & 0b11) as usize],")
    lines.append("        BITS_TO_BASE[((idx >> 2) & 0b11) as usize],")
    lines.append("        BITS_TO_BASE[(idx & 0b11) as usize],")
    lines.append("    ]")
    lines.append("}")
    lines.append("")
    lines.append("/// Static lookup table mapping each 6-bit codon index to its string form.")
    lines.append("pub const CODON_STRINGS: [&str; 64] = [")
    codon_strings = [f'"{idx_to_codon(i)}"' for i in range(64)]
    for i in range(0, 64, 8):
        lines.append("    " + ", ".join(codon_strings[i:i + 8]) + ",")
    lines.append("];")
    lines.append("")
    lines.append("/// One-character amino-acid codes, index = amino-acid index.")
    lines.append("/// Index 20 is the stop codon '*'. There is no selenocysteine entry.")
    lines.append("pub const AA_CHARS: [char; 21] = [")
    lines.append("    " + ", ".join(f"'{aa}'" for aa in AA_ORDER) + ",")
    lines.append("];")
    lines.append("")
    lines.append("/// Convert a one-character amino-acid string to its amino-acid index.")
    lines.append("#[inline]")
    lines.append("pub fn aa_to_idx(aa: &str) -> u8 {")
    lines.append("    try_aa_to_idx(aa).unwrap_or_else(|| panic!(\"unknown amino acid: {}\", aa))")
    lines.append("}")
    lines.append("")
    lines.append("/// Convert a one-character amino-acid string to its amino-acid index, if valid.")
    lines.append("#[inline]")
    lines.append("pub fn try_aa_to_idx(aa: &str) -> Option<u8> {")
    lines.append("    match aa.as_bytes().first()? {")
    for i, aa in enumerate(AA_ORDER):
        lines.append(f"        b'{aa}' => Some({i}),")
    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("/// Amino-acid index for each of the 64 codons.")
    lines.append("pub const AA_OF_CODON: [u8; 64] = [")
    for i in range(0, 64, 8):
        row = ", ".join(f"{aa_of[j]:2d}" for j in range(i, min(i + 8, 64)))
        lines.append(f"    {row},")
    lines.append("];")
    lines.append("")
    lines.append("/// Lists of codon indices belonging to each amino acid.")
    lines.append("pub const CODONS_FOR_AA: [&[u8]; 21] = [")
    for aa in AA_ORDER:
        idxs = sorted(codons_for_aa[aa])
        lines.append("    &[" + ", ".join(str(i) for i in idxs) + f"], // {aa}")
    lines.append("];")
    lines.append("")
    lines.append("// Built-in codon frequency tables")
    lines.append("")

    for t in target_tissues:
        table_name = t.upper().split("_(")[0]
        const_name = f"HUMAN_{table_name}_CODON_TABLE"
        table = {aa: scaled_dfs[aa][t] for aa in scaled_dfs}
        freqs = build_freq_array(table)
        max_freq = build_max_freq(table)

        lines.append(f"pub const {const_name}: CodonTable = CodonTable {{")
        lines.append("    freqs: [")
        for i in range(0, 64, 4):
            row = ", ".join(f"{freqs[j]:.17e}" for j in range(i, min(i + 4, 64)))
            lines.append(f"        {row},")
        lines.append("    ],")
        lines.append("    max_freq: [")
        for i in range(0, 21, 3):
            row = ", ".join(f"{max_freq[j]:.17e}" for j in range(i, min(i + 3, 21)))
            lines.append(f"        {row},")
        lines.append("    ],")
        lines.append("};")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

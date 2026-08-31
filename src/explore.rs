use crate::cai::CaiMode;
use crate::cli::{ExploreOptions, FitOptions, UfitOptions};

use serde_json::Value;
use std::fs::{create_dir_all, read_dir, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// One row in the explore summary.
#[derive(Debug, serde::Serialize)]
struct DesignRecord {
    zone: String,
    mode: String,
    codon_table: String,
    seq_id: String,
    raw_cai: f64,
    scaled_cai: f64,
    arithmetic_cai: f64,
    gc_pct: f64,
    mfe: f64,
    palindrome_score: f64,
    mutate2stop_score: f64,
    file_prefix: String,
    json_path: String,
    command: String,
}

/// Internal description of a single design mode.
struct Mode {
    zone: &'static str,
    name: String,
    kind: ModeKind,
}

enum ModeKind {
    FitGcCai,
    FitPmgc,
    FitGcCaiMcmc { iterations: usize },
    UfitDeopt,
    UfitPllMcmc { iterations: usize },
}

fn mfe_start_codon(opt: &ExploreOptions) -> usize {
    if opt.skip_mfe {
        usize::MAX
    } else {
        0
    }
}

fn base_fit_options(opt: &ExploreOptions, table: &str) -> FitOptions {
    FitOptions {
        fasta: opt.fasta.clone(),
        win_size: opt.win_size,
        rnd_size: 2,
        num_outputs: 1,
        lambda: opt.lambda,
        threads: opt.threads,
        seq_parallel: 1,
        outdir: opt.outdir.clone(),
        is_aa_seq: opt.is_aa,
        is_rna: opt.is_rna,
        homopolymers: opt.homopolymers,
        prefix: None,
        avoid_seqs: opt.avoid_seqs.clone(),
        freq_table_info: table.to_string(),
        gc_cai: false,
        mfe_cai: false,
        no_verbose: opt.no_verbose,
        palindrome: false,
        cmgmp: false,
        pmgc: false,
        maximize_loss: false,
        weight_gc: 1.0,
        weight_cai: 1.0,
        weight_m2s: 1.0,
        weight_palindrome: 1.0,
        mfe_pll_start_codon: mfe_start_codon(opt),
        search_method: "beam".to_string(),
        mcmc_iterations: 1000,
        mcmc_mutations: 5,
        mcmc_accept_prob: 0.5,
        mcmc_temperature: 0.0,
        mcmc_traces: 1,
        mcmc_region: None,
        mfe_method: opt.mfe_method,
        mfe_window: opt.mfe_window,
        weak_head: 0,
        cai_mode: CaiMode::Scaled,
    }
}

fn base_ufit_options(opt: &ExploreOptions, table: &str) -> UfitOptions {
    base_ufit_options_with_pll(opt, table, 0.0, None)
}

fn base_ufit_options_with_pll(
    opt: &ExploreOptions,
    table: &str,
    pll_weight: f64,
    model_path: Option<PathBuf>,
) -> UfitOptions {
    UfitOptions {
        fasta: opt.fasta.clone(),
        win_size: opt.win_size,
        rnd_size: 2,
        num_outputs: 1,
        lambda: opt.lambda,
        weight_gc: 0.5,
        weight_cai: 1.0,
        weight_m2s: 1.0,
        weight_palindrome: 1.0,
        weight_mfe: 0,
        weight_pll: pll_weight,
        model_path: model_path,
        threads: opt.threads,
        seq_parallel: 1,
        outdir: opt.outdir.clone(),
        is_aa_seq: opt.is_aa,
        is_rna: opt.is_rna,
        homopolymers: opt.homopolymers,
        prefix: None,
        avoid_seqs: opt.avoid_seqs.clone(),
        freq_table_info: table.to_string(),
        maximize_loss: true,
        no_verbose: opt.no_verbose,
        mfe_pll_start_codon: mfe_start_codon(opt),
        search_method: "beam".to_string(),
        mcmc_iterations: 1000,
        mcmc_mutations: 5,
        mcmc_accept_prob: 0.5,
        mcmc_temperature: 0.0,
        mcmc_traces: 1,
        mcmc_region: None,
        mfe_method: opt.mfe_method,
        mfe_window: opt.mfe_window,
        weak_head: 0,
        pll_batch_size: 256,
        pll_timeout_ms: 50,
        pll_threads: 0,
        cai_mode: CaiMode::Arithmetic,
    }
}

fn build_schedule(opt: &ExploreOptions) -> Vec<Mode> {
    if opt.pll_weight != 0.0 && opt.model_path.is_none() {
        eprintln!("Error: --pll-weight requires --model-path");
        std::process::exit(1);
    }

    let mut schedule = Vec::new();
    let iters: Vec<usize> = opt
        .intermediate_iters
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| {
            eprintln!(
                "Error: --intermediate-iters must be a comma-separated list of positive integers, got '{}'",
                opt.intermediate_iters
            );
            std::process::exit(1);
        });

    if !opt.no_opt {
        schedule.push(Mode {
            zone: "optimized",
            name: "opt_gc_cai".to_string(),
            kind: ModeKind::FitGcCai,
        });
        schedule.push(Mode {
            zone: "optimized",
            name: "opt_pmgc".to_string(),
            kind: ModeKind::FitPmgc,
        });
    }

    if !opt.no_intermediate {
        for iterations in &iters {
            schedule.push(Mode {
                zone: "intermediate",
                name: format!("inter_gc_cai_mcmc{}", iterations),
                kind: ModeKind::FitGcCaiMcmc {
                    iterations: *iterations,
                },
            });
        }
    }

    if !opt.no_deopt {
        schedule.push(Mode {
            zone: "deoptimized",
            name: "deopt_ufit".to_string(),
            kind: ModeKind::UfitDeopt,
        });
    }

    if opt.pll_weight != 0.0 && !opt.no_intermediate {
        schedule.push(Mode {
            zone: "intermediate",
            name: "inter_ufit_pll_mcmc".to_string(),
            kind: ModeKind::UfitPllMcmc {
                iterations: opt.pll_mcmc_iters,
            },
        });
    }

    if schedule.is_empty() {
        eprintln!("Error: all design zones disabled; nothing to do.");
        std::process::exit(1);
    }

    schedule
}

fn parse_extra_tables(raw: &Option<String>) -> Vec<String> {
    let mut tables = Vec::new();
    if let Some(s) = raw {
        for t in s.split(',') {
            let t = t.trim();
            if t.is_empty() {
                continue;
            }
            if !crate::LEGAL_SHORTS.iter().any(|x| *x == t) {
                eprintln!(
                    "Error: '{}' in --extra-tables is not a valid codon table short name",
                    t
                );
                std::process::exit(1);
            }
            tables.push(t.to_string());
        }
    }
    tables
}

fn make_per_mode_prefix(user_prefix: &Option<String>, table: &str, mode_name: &str) -> String {
    let base = format!("{}_{}", table, mode_name);
    match user_prefix {
        Some(p) => format!("{}_{}", p, base),
        None => format!("explore_{}", base),
    }
}

fn skip_mfe_flag(opt: &ExploreOptions) -> String {
    if opt.skip_mfe {
        " --mfe_pll_start 99999999".to_string()
    } else {
        String::new()
    }
}

fn approximate_command(zone: &str, mode_name: &str, table: &str, opt: &ExploreOptions) -> String {
    let fasta = opt.fasta.display();
    let outdir = opt.outdir.display();
    match zone {
        "optimized" if mode_name == "opt_gc_cai" => {
            format!(
                "cosyn fit -f {} -c {} --gc_cai{} -o {} -l {} -w {} -y {}",
                fasta,
                table,
                skip_mfe_flag(opt),
                outdir,
                opt.lambda,
                opt.win_size,
                opt.homopolymers
            )
        }
        "optimized" if mode_name == "opt_pmgc" => {
            format!(
                "cosyn fit -f {} -c {} --pmgc{} -o {} -l {} -w {} -y {}",
                fasta,
                table,
                skip_mfe_flag(opt),
                outdir,
                opt.lambda,
                opt.win_size,
                opt.homopolymers
            )
        }
        "intermediate" if mode_name.starts_with("inter_gc_cai_mcmc") => {
            let iters = mode_name.strip_prefix("inter_gc_cai_mcmc").unwrap_or("100");
            format!(
                "cosyn fit -f {} -c {} --gc_cai --search-method mcmc --mcmc-iterations {}{} -o {} -l {} -w {} -y {}",
                fasta, table, iters, skip_mfe_flag(opt), outdir, opt.lambda, opt.win_size, opt.homopolymers
            )
        }
        "deoptimized" => {
            format!(
                "cosyn ufit -f {} -c {} --maximize_loss -e 0 -q 0 -x 1.0 -b 0.5{} -o {} -l {} -w {} -y {}",
                fasta,
                table,
                skip_mfe_flag(opt),
                outdir,
                opt.lambda,
                opt.win_size,
                opt.homopolymers
            )
        }
        _ if mode_name.starts_with("inter_ufit_pll_mcmc") => {
            let model = opt
                .model_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            format!(
                "cosyn ufit -f {} -c {} --search-method mcmc --mcmc-iterations {} -e 0 -q {} -j {}{} -o {} -l {} -w {} -y {}",
                fasta,
                table,
                opt.pll_mcmc_iters,
                opt.pll_weight,
                model,
                skip_mfe_flag(opt),
                outdir,
                opt.lambda,
                opt.win_size,
                opt.homopolymers
            )
        }
        _ => format!("# {} {} {}", zone, mode_name, table),
    }
}

fn find_output_json_files(outdir: &Path, prefix: &str, lambda: f64) -> Vec<PathBuf> {
    let prefix_in_filename = format!("{}_out_lambda_{}_", prefix, lambda);
    let mut files = Vec::new();
    if let Ok(entries) = read_dir(outdir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(&prefix_in_filename) && name.ends_with(".json") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

fn parse_result_json(path: &Path) -> Vec<Value> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let val: Value = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Error parsing {}: {}", path.display(), e);
        std::process::exit(1);
    });
    val.get("results")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default()
}

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn as_string(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

pub fn run_explore(opt: &ExploreOptions) -> std::io::Result<()> {
    if !crate::LEGAL_SHORTS.iter().any(|x| *x == opt.codon_table) {
        eprintln!(
            "Error: '{}' is not a valid codon table short name",
            opt.codon_table
        );
        std::process::exit(1);
    }
    if opt.is_aa && opt.is_rna {
        eprintln!("Error: --is-aa and --is-rna cannot be used together");
        std::process::exit(1);
    }

    create_dir_all(&opt.outdir)?;

    let mut tables = vec![opt.codon_table.clone()];
    tables.extend(parse_extra_tables(&opt.extra_tables));

    let schedule = build_schedule(opt);

    let mut records: Vec<DesignRecord> = Vec::new();
    let mut command_lines: Vec<String> = Vec::new();
    let prefix_flag = opt
        .prefix
        .as_ref()
        .map(|p| format!(" -p {}", p))
        .unwrap_or_default();
    command_lines.push(format!(
        "# cosyn explore -f {} -c {} -o {}{}",
        opt.fasta.display(),
        opt.codon_table,
        opt.outdir.display(),
        prefix_flag
    ));
    command_lines.push("# Generated design commands:".to_string());

    for table in &tables {
        for mode in &schedule {
            let per_mode_prefix = make_per_mode_prefix(&opt.prefix, table, &mode.name);
            command_lines.push(format!(
                "# zone={} mode={} table={}",
                mode.zone, mode.name, table
            ));
            let mode_command = approximate_command(mode.zone, &mode.name, table, opt);
            command_lines.push(mode_command.clone());

            match &mode.kind {
                ModeKind::FitGcCai => {
                    let mut fit_opt = base_fit_options(opt, table);
                    fit_opt.gc_cai = true;
                    fit_opt.prefix = Some(per_mode_prefix.clone());
                    crate::run_fit(&fit_opt)?;
                }
                ModeKind::FitPmgc => {
                    let mut fit_opt = base_fit_options(opt, table);
                    fit_opt.pmgc = true;
                    fit_opt.prefix = Some(per_mode_prefix.clone());
                    crate::run_fit(&fit_opt)?;
                }
                ModeKind::FitGcCaiMcmc { iterations } => {
                    let mut fit_opt = base_fit_options(opt, table);
                    fit_opt.gc_cai = true;
                    fit_opt.search_method = "mcmc".to_string();
                    fit_opt.mcmc_iterations = *iterations;
                    fit_opt.prefix = Some(per_mode_prefix.clone());
                    crate::run_fit(&fit_opt)?;
                }
                ModeKind::UfitDeopt => {
                    let mut ufit_opt = base_ufit_options(opt, table);
                    ufit_opt.prefix = Some(per_mode_prefix.clone());
                    crate::run_ufit(&ufit_opt)?;
                }
                ModeKind::UfitPllMcmc { iterations } => {
                    let mut ufit_opt = base_ufit_options_with_pll(
                        opt,
                        table,
                        opt.pll_weight,
                        opt.model_path.clone(),
                    );
                    ufit_opt.search_method = "mcmc".to_string();
                    ufit_opt.mcmc_iterations = *iterations;
                    ufit_opt.prefix = Some(per_mode_prefix.clone());
                    crate::run_ufit(&ufit_opt)?;
                }
            }

            let json_files = find_output_json_files(&opt.outdir, &per_mode_prefix, opt.lambda);
            if json_files.is_empty() {
                eprintln!(
                    "Warning: no output JSON found for {} / {}",
                    table, mode.name
                );
                continue;
            }

            for json_path in &json_files {
                for result in parse_result_json(json_path) {
                    records.push(DesignRecord {
                        zone: mode.zone.to_string(),
                        mode: mode.name.to_string(),
                        codon_table: table.clone(),
                        seq_id: as_string(&result["seq_id"]),
                        raw_cai: as_f64(&result["raw_cai"]),
                        scaled_cai: as_f64(&result["scaled_cai"]),
                        arithmetic_cai: as_f64(&result["arithmetic_cai"]),
                        gc_pct: as_f64(&result["GC%"]),
                        mfe: as_f64(&result["mfe"]),
                        palindrome_score: as_f64(&result["palindrome_score"]),
                        mutate2stop_score: as_f64(&result["mutate2stop_score"]),
                        file_prefix: per_mode_prefix.clone(),
                        json_path: json_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string(),
                        command: mode_command.clone(),
                    });
                }
            }
        }
    }

    write_summary_csv(&opt.outdir, &records)?;
    write_summary_json(&opt.outdir, &records, opt)?;
    write_combined_fasta(&opt.outdir, &records)?;
    write_commands_txt(&opt.outdir, &command_lines)?;

    let table_count = tables.len();
    let mode_count = schedule.len();
    println!(
        "Explore finished: {} codon table(s) x {} mode(s) = {} designs written to {}",
        table_count,
        mode_count,
        records.len(),
        opt.outdir.display()
    );

    Ok(())
}

fn write_summary_csv(outdir: &Path, records: &[DesignRecord]) -> std::io::Result<()> {
    let path = outdir.join("explore_summary.csv");
    let handle = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(handle);
    writeln!(
        writer,
        "zone,mode,codon_table,seq_id,raw_cai,scaled_cai,arithmetic_cai,GC%,mfe,palindrome_score,mutate2stop_score,file_prefix,json_path,command"
    )?;
    for r in records {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.zone,
            r.mode,
            r.codon_table,
            quote_if_needed(&r.seq_id),
            r.raw_cai,
            r.scaled_cai,
            r.arithmetic_cai,
            r.gc_pct,
            r.mfe,
            r.palindrome_score,
            r.mutate2stop_score,
            r.file_prefix,
            r.json_path,
            quote_if_needed(&r.command)
        )?;
    }
    Ok(())
}

fn quote_if_needed(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn write_summary_json(
    outdir: &Path,
    records: &[DesignRecord],
    opt: &ExploreOptions,
) -> std::io::Result<()> {
    let path = outdir.join("explore_summary.json");
    let handle = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(handle);
    let mut val = serde_json::json!({ "results": records });
    if let Some(obj) = val.as_object_mut() {
        obj.insert(
            "params".to_string(),
            serde_json::to_value(opt)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
        );
    }
    writer.write_all(serde_json::to_string_pretty(&val)?.as_bytes())?;
    Ok(())
}

fn write_combined_fasta(outdir: &Path, records: &[DesignRecord]) -> std::io::Result<()> {
    let path = outdir.join("explore_all.fasta");
    let handle = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(handle);

    for r in records {
        // Read the CDS from the per-mode JSON so we always have the exact sequence.
        let json_path = outdir.join(&r.json_path);
        let cds = if let Ok(content) = std::fs::read_to_string(&json_path) {
            if let Ok(val) = serde_json::from_str::<Value>(&content) {
                val.get("results")
                    .and_then(|arr| arr.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .find(|entry| as_string(&entry["seq_id"]) == r.seq_id)
                    })
                    .and_then(|entry| entry["optimized_cds"].as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        if cds.is_empty() {
            continue;
        }
        let clean_id = r.seq_id.trim_start_matches('>');
        writeln!(
            writer,
            ">{} table={} zone={} mode={} raw_cai={:.4} GC={:.2} mfe={:.2}",
            clean_id, r.codon_table, r.zone, r.mode, r.raw_cai, r.gc_pct, r.mfe
        )?;
        writeln!(writer, "{}", cds)?;
    }
    Ok(())
}

fn write_commands_txt(outdir: &Path, lines: &[String]) -> std::io::Result<()> {
    let path = outdir.join("explore_commands.txt");
    let handle = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(handle);
    for line in lines {
        writeln!(writer, "{}", line)?;
    }
    Ok(())
}

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "LinearFold Rust port")]
struct Args {
    /// RNA sequence (A, C, G, U/T)
    #[arg(short, long)]
    seq: String,

    /// Maximum base-pair span (0 = unlimited)
    #[arg(short, long, default_value = "0")]
    window: usize,
}

fn main() {
    let args = Args::parse();
    let seq = args.seq.to_uppercase().replace('T', "U");
    let max_pair_dist = if args.window == 0 {
        None
    } else {
        Some(args.window)
    };
    let (mfe, structure) =
        linearfold_rust::rna_linear_mfe_with_options(&seq, 100, true, max_pair_dist);
    println!("sequence:  {}", seq);
    println!("structure: {}", structure);
    println!("MFE:       {:.2}", mfe);
}

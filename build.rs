fn main() {
    // The `pll` Cargo feature controls whether CodonTransformer PLL support is
    // compiled in. When disabled, we skip building/linking the C++ libtorch
    // wrapper entirely, so cosyn can be built on systems without libtorch/g++.
    let pll_enabled = std::env::var("CARGO_FEATURE_PLL").is_ok();

    if pll_enabled {
        // Detect CPU-only mode from env var set by Makefile's build_cpu / release_cpu targets.
        let cpu_only = std::env::var("CODON_CPU_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Link the appropriate codon_pll static library.
        println!("cargo:rustc-link-search=lib/");
        if cpu_only {
            println!("cargo:rerun-if-changed=lib/libcodon_pll_cpu.a");
            println!("cargo:rustc-link-lib=static=codon_pll_cpu");
            println!("cargo:warning=CODON_CPU_ONLY=1 — linking libcodon_pll_cpu.a");
        } else {
            println!("cargo:rerun-if-changed=lib/libcodon_pll.a");
            println!("cargo:rustc-link-lib=static=codon_pll");
        }

        // libstdc++ is required because codon_pll.cpp is compiled with g++.
        println!("cargo:rustc-link-lib=dylib=stdc++");

        // libtorch is dynamically linked; add its search path and rpath.
        if let Ok(libtorch) = std::env::var("LIBTORCH") {
            println!("cargo:rustc-link-search={}/lib", libtorch);
            println!("cargo:rustc-link-lib=dylib=torch");
            println!("cargo:rustc-link-lib=dylib=torch_cpu");
            println!("cargo:rustc-link-lib=dylib=c10");

            // Auto-detect whether the installed libtorch includes CUDA libraries.
            // If libtorch_cuda.so exists, force-link it with --no-as-needed
            // (the linker would otherwise strip it because our binary does not
            // directly reference its symbols; libtorch loads it via dlopen).
            // Skip CUDA linking when CODON_CPU_ONLY=1 is set.
            let cuda_so = std::path::Path::new(&libtorch).join("lib/libtorch_cuda.so");
            if !cpu_only && cuda_so.exists() {
                println!(
                    "cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,-lc10_cuda,--as-needed"
                );
            } else if cpu_only {
                println!("cargo:warning=CODON_CPU_ONLY=1 — skipping CUDA linking");
            }

            // Set rpath so the binary can find libtorch at runtime
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", libtorch);
        }
    } else {
        println!("cargo:warning=PLL feature disabled — building cosyn without CodonTransformer PLL support.");
    }
}

use std::env;

fn main() {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib", cuda_home);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    }
    println!("cargo:rustc-link-lib=cudart");
}

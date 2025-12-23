{
  description = "Development shell with maturin and CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      cudaPkgs = pkgs.cudaPackages_12;
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3
          uv

          rustc
          cargo
          maturin

          cudaPkgs.cudatoolkit
          cudaPkgs.cudnn
          cudaPkgs.libcublas
          cudaPkgs.libcufft
          cudaPkgs.libcurand
          cudaPkgs.libcusparse
          cudaPkgs.libcusolver
          cudaPkgs.nccl
          cudaPkgs.tensorrt
          cudaPkgs.tensorrt.lib

          libGL
          libGLU
          xorg.libX11
          xorg.libXext
          xorg.libXrender

          stdenv.cc.cc.lib
          stdenv.cc.libc
          stdenv.cc.libc.dev
          glibc.dev

          stdenv.cc  # Use wrapped compiler instead of gcc directly
          cmake
          pkg-config

          openssl
          cacert
        ];

        shellHook = ''
          export CUDA_PATH=${cudaPkgs.cudatoolkit}
          export CUDA_HOME=$CUDA_PATH
          export CUDA_ROOT=$CUDA_PATH

          export TENSORRT_PATH=${cudaPkgs.tensorrt.lib}

          export CUDNN_PATH=${cudaPkgs.cudnn.lib}
          if [ -d "/run/opengl-driver/lib" ]; then
            export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$TENSORRT_PATH/lib:$CUDNN_PATH/lib:/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          else
            export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$TENSORRT_PATH/lib:$CUDNN_PATH/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          fi

          # For torch.compile C++ backend - use stdenv.cc which has proper include paths
          export CC=${pkgs.stdenv.cc}/bin/cc
          export CXX=${pkgs.stdenv.cc}/bin/c++
          
          # Create a wrapper script for c++ that includes NixOS system headers
          # This is needed because PyTorch's inductor invokes c++ directly without
          # respecting CPATH or NIX_CFLAGS_COMPILE for #include_next directives
          mkdir -p /tmp/nix-cc-wrapper
          cat > /tmp/nix-cc-wrapper/c++ << 'WRAPPER_EOF'
#!/bin/bash
exec ${pkgs.stdenv.cc}/bin/c++ -isystem ${pkgs.stdenv.cc.libc.dev}/include "$@"
WRAPPER_EOF
          chmod +x /tmp/nix-cc-wrapper/c++
          cp /tmp/nix-cc-wrapper/c++ /tmp/nix-cc-wrapper/g++
          export PATH=/tmp/nix-cc-wrapper:$PATH
          
          # Also set these for good measure
          export C_INCLUDE_PATH=${pkgs.stdenv.cc.libc.dev}/include:$C_INCLUDE_PATH
          export CPLUS_INCLUDE_PATH=${pkgs.stdenv.cc.libc.dev}/include:$CPLUS_INCLUDE_PATH
          export CPATH=${pkgs.stdenv.cc.libc.dev}/include:$CPATH
          
          unset NIX_ENFORCE_NO_NATIVE
          export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

          export PATH=$CUDA_PATH/bin:$PATH
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

          # For ort crate to find OpenSSL
          export OPENSSL_DIR=${pkgs.openssl.dev}
          export OPENSSL_LIB_DIR=${pkgs.openssl.out}/lib
          export OPENSSL_INCLUDE_DIR=${pkgs.openssl.dev}/include
          export PKG_CONFIG_PATH=${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH
        '';
      };
    });
}

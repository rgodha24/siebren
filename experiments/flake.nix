{
  description = "Experiment shell with Python 3.12 and CUDA";

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
          python312
          uv

          rustc
          cargo

          cudaPkgs.cudatoolkit

          stdenv.cc
          stdenv.cc.cc.lib
          cmake
          pkg-config

          openssl
          cacert
        ];

        shellHook = ''
          export CUDA_PATH=${cudaPkgs.cudatoolkit}
          export CUDA_HOME=$CUDA_PATH
          export CUDA_ROOT=$CUDA_PATH

          if [ -d "/run/opengl-driver/lib" ]; then
            export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          else
            export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          fi

          export CC=${pkgs.stdenv.cc}/bin/cc
          export CXX=${pkgs.stdenv.cc}/bin/c++

          export PATH=$CUDA_PATH/bin:$PATH
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

          export OPENSSL_DIR=${pkgs.openssl.dev}
          export OPENSSL_LIB_DIR=${pkgs.openssl.out}/lib
          export OPENSSL_INCLUDE_DIR=${pkgs.openssl.dev}/include
          export PKG_CONFIG_PATH=${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH
        '';
      };
    });
}

{
  description = "dev shell with python and optional cuda support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nix-gl-host.url = "github:numtide/nix-gl-host/5269b233f83880a0b433eafe026f0bc0d8f1a4a9";
    flake-utils.url = "github:numtide/flake-utils/11707dc2f618dd54ca8739b309ec4fc024de578b";
  };

  outputs =
    {
      self,
      nix-gl-host,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        isLinux = nixpkgs.lib.hasInfix "linux" system;

        # Read Python version from pyproject.toml, not the cleanest way to do it
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        pythonVersion = builtins.match ".*([0-9]+\\.[0-9]+).*" pyproject.project.requires-python;
        pythonVersionAttr =
          "python" + builtins.replaceStrings [ "." ] [ "" ] (builtins.head pythonVersion) + "Full";

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = isLinux;
          };
        };

        # Base packages available on all systems
        basePackages = with pkgs; [
          gcc
          uv
          ruff
          (builtins.getAttr pythonVersionAttr pkgs)
        ];

        # CUDA packages only on Linux
        cudaPackages = nixpkgs.lib.optionals isLinux [
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          cudaPackages.cuda_cupti
          cudaPackages.cuda_nvrtc
          cudaPackages.cuda_nvtx
          cudaPackages.cudnn
          cudaPackages.libcublas
          cudaPackages.libcufft
          cudaPackages.libcurand
          cudaPackages.libcusolver
          cudaPackages.libcusparse
          cudaPackages.libnvjitlink
          cudaPackages.nccl
          nix-gl-host.defaultPackage.${system}
        ];

        allPackages = basePackages ++ cudaPackages;

        # Shell hook with conditional CUDA setup
        shellHook =
          ''
            # Set up the virtual environment
            export UV_PROJECT_ENVIRONMENT=.nix-venv
            uv sync
            source .nix-venv/bin/activate
          ''
          + nixpkgs.lib.optionalString isLinux ''

            # Set up the CUDA environment (Linux only)
            export LD_LIBRARY_PATH=$(nixglhost -p):$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${nixpkgs.lib.makeLibraryPath allPackages}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages = allPackages;
          inherit shellHook;
        };
      }
    );
}

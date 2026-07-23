{
  description = "talking-portrait.cpp: C++/GGML port of FLOAT (audio-driven talking portrait)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        # CPU build (presets: debug, release)
        default = pkgs.mkShell {
          packages = with pkgs; [ cmake ninja git ffmpeg ];
        };

        # Vulkan build (preset: vulkan) — ggml needs headers, loader and glslc
        vulkan = pkgs.mkShell {
          packages = with pkgs; [
            cmake ninja git ffmpeg
            shaderc vulkan-headers vulkan-loader vulkan-tools
          ];
        };

        # Fuzz build (preset: fuzz) — clang stdenv for libFuzzer
        fuzz = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          packages = with pkgs; [ cmake ninja git ];
        };
      });

      packages = forAllSystems (pkgs:
        let
          hasSubmodule = builtins.pathExists (self + "/ggml/CMakeLists.txt");
        in {
          default =
            if !hasSubmodule then
              throw "ggml submodule missing from flake source — build with: nix build '.?submodules=1'"
            else
              pkgs.stdenv.mkDerivation {
                pname = "talking-portrait";
                version = self.shortRev or self.dirtyShortRev or "dev";
                src = self;

                nativeBuildInputs = with pkgs; [ cmake ninja git makeWrapper ];

                cmakeFlags = [
                  "-DGGML_LLAMAFILE=ON"
                  "-DTP_BUILD_TESTS=OFF"
                  # static ggml: no build-tree RPATHs to scrub, nothing to install but the tools
                  "-DBUILD_SHARED_LIBS=OFF"
                ];

                # no install() rules upstream; binaries land in the build root
                installPhase = ''
                  runHook preInstall
                  install -Dm755 -t $out/bin tp-cli tp-parity tp-bench tp-allocproof
                  runHook postInstall
                '';

                # tp-cli spawns ffmpeg for --out *.mp4
                postFixup = ''
                  wrapProgram $out/bin/tp-cli \
                    --prefix PATH : ${nixpkgs.lib.makeBinPath [ pkgs.ffmpeg ]}
                '';

                meta = {
                  mainProgram = "tp-cli";
                  license = nixpkgs.lib.licenses.mit;
                };
              };
        });
    };
}

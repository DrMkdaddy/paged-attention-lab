{
  description = "Paged attention: block KV cache with block tables and copy-on-write";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system} system);
    in {
      devShells = forAll (pkgs: _: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [ ps.numpy ]))
            pkgs.python3Packages.pytest
          ];
          shellHook = ''
            echo "paged-attention-lab: python3 tests/test_paged_attention.py"
          '';
        };
      });

      checks = forAll (pkgs: _: {
        tests = pkgs.runCommand "paged-attention-tests" {
          buildInputs = [ (pkgs.python3.withPackages (ps: [ ps.numpy ])) ];
        } ''
          cp -r ${self} work && chmod -R u+w work && cd work
          python3 tests/test_paged_attention.py
          touch $out
        '';
      });

      packages = forAll (pkgs: _: {
        default = pkgs.runCommand "paged-attention-lab" { } ''
          cp -r ${self} $out
        '';
      });
    };
}

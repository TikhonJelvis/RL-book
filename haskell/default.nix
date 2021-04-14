{ sources ? import nix/sources.nix {}
, compiler-nix-name ? "ghc884"
}:
let
  haskellNix = import sources."haskell.nix" {};
  pkgs = import sources.nixpkgs haskellNix.nixpkgsArgs;
in
pkgs.haskell-nix.project {
  src = pkgs.haskell-nix.haskellLib.cleanGit {
    name = "rl-book";
    src = ./.;
  };
  inherit compiler-nix-name;
}

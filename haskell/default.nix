{ sources ? import nix/sources.nix
, pkgs ? import sources.nixpkgs {}
}:
pkgs.haskellPackages.developPackage {
  name = "rl-book";
  root = (pkgs.lib.cleanSourceWith {
    src = ./.;
    filter = path: type:
      let
        name = baseNameOf (toString path);
        ignored = ["dist" "dist-newstyle"];
      in
        builtins.all (ignored-file: name != ignored-file) ignored &&
        !pkgs.lib.hasPrefix ".ghc.environment" name &&
        pkgs.lib.cleanSourceFilter path type;
  }).outPath;

  overrides = new: old: {
    monad-bayes = pkgs.haskell.lib.unmarkBroken (pkgs.haskell.lib.doJailbreak old.monad-bayes);
  };

  # Disable "smart" Nix-shell detection because it is impure (depends
  # on the IN_NIX_SHELL environment variable), which can cause
  # hard-to-debug Nix issues.
  #
  # Instead, we have an explicit shell.nix with extra shell-specific
  # settings.
  returnShellEnv = false;
}

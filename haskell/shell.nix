{ sources ? import nix/sources.nix {}
,  pkgs ? import sources.nixpkgs {}
}:
pkgs.lib.overrideDerivation (import ./. { inherit sources pkgs; }).env (old: {
  nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.cabal-install ];
})

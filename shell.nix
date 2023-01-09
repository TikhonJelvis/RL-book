{ sources ? import nix/sources.nix
, pkgs ? import sources.nixpkgs {}
, python-version ? "3.8"
}:
let
  versions = {
    "3.6" = pkgs.python36;
    "3.7" = pkgs.python37;
    "3.8" = pkgs.python38;
    "3.9" = pkgs.python39;
    "3.10" = pkgs.python310;
  };
in
import ./. { python = versions."${python-version}"; }

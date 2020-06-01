{ pkgs ? import <nixpkgs> {} }:

let
  texlive = pkgs.texlive.combine {
    inherit (pkgs.texlive)
      scheme-medium
      footmisc
      titling
      xpatch
      noto;
  };
  fonts = pkgs.makeFontsConf {
    fontDirectories = [ pkgs.eb-garamond pkgs.tex-gyre.pagella ];
  };
in
pkgs.stdenv.mkDerivation {
  name = "RL-book";
  src = ./.;

  buildInputs = [
    texlive

    pkgs.fontconfig
    pkgs.graphviz
    pkgs.pandoc
    pkgs.watchexec

    pkgs.python38
  ];

  FONTCONFIG_FILE = fonts;
}

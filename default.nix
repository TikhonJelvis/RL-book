{ pkgs ? import <nixpkgs> {}
, python ? pkgs.python38
}:

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

  pythonWithPackages = python.withPackages (ps:
    with ps; [ graphviz ipython jedi jupyter matplotlib mypy numpy pandas pylint scipy ]);

  system-packages =
    if pkgs.stdenv.isDarwin
    then [ python pkgs.fswatch ]
    else [ pythonWithPackages ];
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
  ] ++ system-packages;

  FONTCONFIG_FILE = fonts;
}

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

  pythonDependencies = ps: with ps; [ graphviz ipython jedi jupyter matplotlib numpy pandas scipy ];
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

    (python.withPackages (ps: pythonDependencies ps))
  ];

  FONTCONFIG_FILE = fonts;
}

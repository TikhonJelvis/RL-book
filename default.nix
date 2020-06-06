{ pkgs ? import <nixpkgs> {}
, basePython ? pkgs.python38
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

  python = if pkgs.stdenv.isDarwin
           then basePython
           else basePython.withPackages (ps: pythonDependencies ps);
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

    python
  ];

  FONTCONFIG_FILE = fonts;
}

{ pkgs ? import <nixpkgs> {}
, python ? pkgs.python38
}:

let
  tex-packages = {
    inherit (pkgs.texlive)
      scheme-medium
      footmisc
      titling
      xpatch
      noto;
  };

  python-packages = ps: with ps;
    [ graphviz
      ipython
      jedi
      jupyter
      matplotlib
      mypy
      numpy
      pandas
      pylint
      scipy
    ];

  # Applications and utilties for buidling the book
  packages = with pkgs;
    [ fontconfig
      graphviz
      pandoc
      watchexec

      haskellPackages.pandoc-crossref

      (texlive.combine tex-packages)
    ];

  fonts = with pkgs;
    [ eb-garamond
      tex-gyre.pagella
    ];

  pythonWithPackages = python.withPackages python-packages;

  system-packages =
    if pkgs.stdenv.isDarwin
    then [ python pkgs.fswatch ]
    else [ pythonWithPackages ];
in
pkgs.stdenv.mkDerivation {
  name = "RL-book";
  src = ./.;

  buildInputs = with pkgs; packages ++ system-packages;

  FONTCONFIG_FILE = pkgs.makeFontsConf {
    fontDirectories = fonts;
  };
}

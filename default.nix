{ sources ? import nix/sources.nix
, pkgs ? import sources.nixpkgs {}
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
    [ # Libraries
      graphviz
      matplotlib
      numpy
      pandas
      scipy

      # Tools
      ipython
      jedi
      jupyter
      pytest

      # Checkers
      flake8
      mypy
      pylint
    ];

  ghc = "ghc884";

  pandoc-include-code = import sources.pandoc-include-code {
    inherit pkgs;
    compiler = ghc;
  };

  # Applications and utilties for buidling the book
  packages = with pkgs;
    [ fontconfig
      graphviz
      pandoc
      watchexec

      haskell.packages.${ghc}.pandoc-crossref
      pandoc-include-code

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
    else [ pythonWithPackages pkgs.python-language-server ];
in
pkgs.stdenv.mkDerivation {
  name = "RL-book";
  src = ./.;

  buildInputs = with pkgs; packages ++ system-packages;

  FONTCONFIG_FILE = pkgs.makeFontsConf {
    fontDirectories = fonts;
  };
}

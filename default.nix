{ sources ? import nix/sources.nix
, pkgs ? import sources.nixpkgs {}
, pkgs-unstable ? import sources.nixpkgs-unstable {}
, python ? pkgs.python38
}:

let
  tex-packages = {
    inherit (pkgs.texlive)
      scheme-medium

      appendix
      footmisc
      imakeidx
      noto
      titling
      xpatch ;
  };

  python-packages = ps: with ps;
    [ # Libraries
      graphviz
      matplotlib
      numpy
      pandas
      scipy
      fire

      # Tools
      black
      ipython
      jedi
      jupyter
      pytest

      # Checkers
      flake8
      mypy
      pylint
    ];

  # Applications and utilties for buidling the book
  packages = with pkgs;
    [ fontconfig
      graphviz
      watchexec

      # we need a newer version of pandoc to fix bibliography sorting
      #
      # pandoc-crossref needs to be built with the matching version of
      # pandoc
      pkgs-unstable.pandoc
      pkgs-unstable.haskellPackages.pandoc-crossref

      (texlive.combine tex-packages)
    ];

  fonts = with pkgs;
    [ eb-garamond
      tex-gyre.pagella
      dejavu_fonts
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

  DEJA_VU_DIRECTORY = "${pkgs.dejavu_fonts}/share/fonts/truetype/";

  # Should be set to an absolute path to the latex directory that's in
  # the same directory as *this file*
  #
  # Note: trailing comma (:) is important! Without it, LaTeX won't
  # find standard classes like article.cls.
  TEXINPUTS = "${toString ./latex}:";
}

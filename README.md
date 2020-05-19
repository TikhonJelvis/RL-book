# RL Book

Basic setup for working with Pandoc and TeX.

## Installation

To work on the book, you need to [install Nix][install]. Do this by running:

```
> curl -L https://nixos.org/nix/install | sh
```

then following the instructions from the script.

Once you have Nix installed, run `nix-shell` to get access to Pandoc, LaTeX and all the other tools you need. The first time you run `nix-shell` will take a while to finish as it downloads and installs all the packages you need.

[install]: https://nixos.org/download.html

## Development

Once inside the Nix shell, you'll have access to Pandoc and you'll be able to generate PDFs with XeTeX. The `to-pdf` script does this for a single Markdown file:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf example
```

(Note that it's `example` and not `example.md`.)

Over time, I'm going to add more functionality to this system, depending on exactly what we need. I'll probably set up a way to generate PDF/Word/etc documents using `nix-build` as well.

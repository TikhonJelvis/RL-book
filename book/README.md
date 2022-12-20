# Book Contents

This directory contains all of the chapters in the book. Each chapter has its own directory, with a corresponding `.md` file (eg Chapter 4 has `chapter4/chapter4.md`) as well as any additional files like images.

To be included in the book, each chapter needs to be added to the [`structure`][1] file in the top level of the repo.

[1]: ../structure

# Compiling the PDF

Basic setup for working with Pandoc and TeX.

## Installation

To work on the book, you need to [install Nix][install].

### Set up the environment

On macOS, first install the XCode command-line tools:

```
xcode-select --install
```

the install Nix with:


```
sh <(curl https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume
```

On Linux, install Nix with:

```
curl -L https://nixos.org/nix/install | sh
```

### Nix Shells

Once you have Nix installed, run `nix-shell` to get access to Pandoc, LaTeX and all the other tools you need. The first time you run `nix-shell` will take a while to finish as it downloads and installs all the packages you need.

[install]: https://nixos.org/download.html

## Generating PDFs

Once inside the Nix shell, you'll have access to Pandoc and you'll be able to generate PDFs with XeTeX. The `to-pdf` script can do this for a single chapter in the [`book`][book] directory:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf chapter0
Converting book/chapter0/chapter0.md to book/chapter0/chapter0.pdf
```

[book]: ./book

You can also generate the entire book to a file called `book.pdf`:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf
Combining
book/chapter0/chapter0.md
book/chapter2/chapter2.md
book/chapter3/chapter3.md
book/chapter4/chapter4.md
book/chapter5/chapter5.md
into book.pdf
```

Note that this can take a little while (10–20 seconds for chapters 0–5).

### Index

We can generate an index with the `--index` flag:

``` shell
bin/to-pdf chapter1 --index
```

This requires running `xelatex` twice, so it'll take longer to generate the PDF. Index terms [are defined within the text with the `\index` command][index-command].

[index-command]: INDEXING.md

### Cross-references

We can define labels for chapters and headings:

```
# Overview {#sec:overview}

## Learning Reinforcement Learning {#sec:learning-rl}
```

Because of limitations with the system I'm using for managing internal references, labels for sections *and* chapters always have to start with `sec:`.

Once you have defined a label for a section or chapter, you can reference its number as follows:

```
Take a look at Chapter [-@sec:mdp].
```

> Take a look at Chapter 3.

For sections, you can also use:

```
Take a look at [@sec:learning-rl].
```

> Take a look at sec. 1.

(The `[-@sec:foo]` syntax drops the "sec. " text.)

For references across chapters to render correctly, you have to compile the entire book PDF (following the instructions above).

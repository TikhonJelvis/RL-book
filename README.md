# RL Book

Basic setup for working with Pandoc and TeX.

To work on the book, you need to [install Nix][install] and then run `nix-shell`. The first time you do this, it will take a while to download and install all the packages it needs.

Once inside the Nix shell, you'll have access to Pandoc and you'll be able to generate PDFs with XeTeX. The `to-pdf` script does this for a single Markdown file:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf example
```

(Note that it's `example` and not `example.md`.)

Over time, I'm going to add more functionality to this system, depending on exactly what we need. I'll probably set up a way to generate PDF/Word/etc documents using `nix-build` as well.

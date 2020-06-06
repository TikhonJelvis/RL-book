# RL Book Setup

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

## Development

Once inside the Nix shell, you'll have access to Pandoc and you'll be able to generate PDFs with XeTeX. The `to-pdf` script does this for a single Markdown file:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf example
```

(Note that it's `example` and not `example.md`.)

Over time, I'm going to add more functionality to this system, depending on exactly what we need. I'll probably set up a way to generate PDF/Word/etc documents using `nix-build` as well.

## Working with Python and venv

We can manage our Python dependencies with a venv.

First, create a venv *from inside a Nix shell*:

```
> nix-shell
[nix-shell:~/Documents/RL-book]$ python -m venv .venv
```

Then, each time you're working on this project, make sure to *activate* the venv:

```
> source .venv/bin/activate
```

(This can now be done even outside a Nix shell.)

Once the venv is activated, you should see a `(.venv)` in your shell prompt:

```
(.venv) RL-book:RL-book>
```

Now you can use `pip` to install dependencies inside the venv:

```
(.venv) RL-book:RL-book> pip install matplotlib
```

To make this reproducible, we can save the libraries to a `requirements.txt` file:

```
(.venv) RL-book:RL-book>pip freeze > requirements.txt
```

Then, when somebody is starting, they can install every Python package they need using:

```
(.venv) RL-book:RL-book>pip install -r requirements.txt
```

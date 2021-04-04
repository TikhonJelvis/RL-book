{ compiler-nix-name ? "ghc884" }:

(import ./. { inherit compiler-nix-name; }).shellFor {
  tools = {
    cabal = "3.2.0.0";
    brittany = "0.13.1.1";
    hindent = "5.3.2";
  };
  exactDeps = true;
}

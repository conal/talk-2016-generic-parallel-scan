[reification-rules]: https://github.com/conal/reification-rules

## Examples for scan talk

I based this project on [reify-core-examples](https://github.com/conal/reify-core-examples).

Reification and circuit generation happen via the GHC option `-fplugin=ReificationRules.Plugin` in the .cabal file and the `go` function from `ReificationRules.Run` in Examples.hs.

## Try it out

You'll need [graphviz](http://www.graphviz.org/) for rendering the circuit diagrams. For instance, "brew install graphviz" on Mac OS, if you have [homebrew](http://brew.sh/) installed.

You'll also need GHC 8, so if you don't already have it, do

    stack setup 8.0.1

With these preliminaries done, build the project and run the examples:

    stack build && stack exec examples

If several examples are enabled, the `build` step may take a few minutes. If you're on Mac OS (or another OS that supports "open") and if everything is working, you'll see one or more displayed PDFs. The PDF gets saved in out/.

To enable/disable test examples and add new ones, edit Examples.hs.

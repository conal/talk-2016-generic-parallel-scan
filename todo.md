---
title: *To do* items for revised *Generic parallel scan*
subst: ["&&& △","*** ×","||| ▽","+++ +","|- ⊢","<~ ⤺","k (↝)","op (⊙)","--> ⇨","+-> ➔",":*: ✖",":+: ➕",":->? ⤔","Unit ()","R ℝ","Unit 𝟙",":==> ⤇"]
---

## To do

*   Complexity --- Master Theorem
*   Conclusion
*   Statistics summary
*   Consider a picture with 2D layout for parallel scan, akin to FFT.
    *   Horizontal arrows to show the scan paths.
    *   Add a result column for the sub-totals.
        The extra scan is vertical.
    *   Maybe also a picture for the linear sequential algorithm, showing dependency chains.
    *   Do the circuit pictures give this information?
        If not, could they?
    *   Wide vs tall.
    *   Maybe my circuit diagrams would work fine.


## Done

*   Generics / tinker toys.
    Show at least some of the `LScan` instances, including product and (especially) composition.
*   Use `LPow` and `RPow` from *Generic FFT*.
*   `LVec` and `RVec`.
*   `Bush`
*   Maybe at first do simple inclusive scan for the pictures.
    We'll see later why use exclusive+fold.
*   Do functor product before composition:
    *   Show `Pair :.: LVec N8` and `LVec N8 :.: Pair`.
        Already defined and rendered in `reify-core-examples`.
*   `lproducts` with `LBin`, `RBin`, and `Bush`.
*   Change `LScan` to use `c a -> (c :*: Id) a`.
    *   Oh. I've already done it in `ShapedTypes.ScanF`.
    *   Maybe use an infix operator in place of `And1`.

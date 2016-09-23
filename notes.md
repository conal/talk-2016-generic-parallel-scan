% High-level algorithm design for reschedulable computation, Part 1 -- Understanding parallel scan
% Conal Elliott, Tabula
% October 8, 2013

# Abstract

Mainstream programming languages are based on a sequential notion of computation pioneered by John von Neumann and others and are thereby deeply biased toward *sequential* execution.
This bias persists even in the presence of multi-threading, because sequentiality is much lighter weight notationally and much simpler semantically in these languages.
For the age of massively parallel computation, we will want to remove sequential bias from both the hardware platforms on which we execute *and* the software languages and techniques in which we think about and express algorithms.
We call these platforms, languages, and techniques "*reschedulable*" to reflect that scheduling of operations is automated, rather than specified explicitly.
In other words, low- and high-level computation descriptions include *what* and *how*, leaving a great deal of flexibility about *when*.

The Spacetime architecture and the Stylus toolchain provide an execution platform and hardware-level "programming" interface for reschedulable computing but lacks the productivity of a modern, high-level programming language.
"von Neumann" (i.e., imperative) languages help with productivity, but over-sequentialize.
In contrast, purely functional programming enables elegant specification of algorithms without accidental sequentialization and thus holds great promise for reschedulable programming.

This talk will give a flavor of purely functional, rescheduling-friendly programming by means of the example of parallel prefix computation, also known as "scan".

We'll start with a simple and precise scan specification that leads directly to an algorithm performing quadratic work and hence quadratic time in a sequential implementation.
A simple observation leads to a familiar sequential algorithm performing linear work.
This version, however, does not lend itself to parallel execution.
Next, a simple divide-and-conquer idea leads to a functional algorithm that can execute in $O (\log n)$ time and $O (n \log n)$ work, given sufficient computational resources.
Playing with variations and poking at asymmetries, we then see a beautiful generalization that leads to a linear work algorithm while retaining logarithmic time.

# Misc thoughts on presentation

Parallel prefix / scan:

*   Show a few formulations.
*   Describe how I grappled with what's going on:
    *   What's the index bit fiddling about?
    *   Hunch about binary trees.
        *   Habitual top-down assumption leads to work-inefficient algorithm.
        *   MSB: top-down trees. LSB: bottom-up trees.
    *   ...
    *   Conclusions/reflections:
        *   We can approach software elegantly and rigorously.
        *   Avoid operationalisms such as explicit scheduling (sequential or parallel).
            *    What makes scheduling matter semantically? Mutation.
            *    Instead, use order-independent foundation: math / pure functional programming.
            *    Data dependencies remain, constraining but not dictating a schedule.
            *    Minimize critical path.
        *   There are some common "control" patterns, such as fold, unfold, and scan.
            Defined "algebraically": equational properties ("laws").
        *   There are some common "data" patterns, such as products, sum, and composition.
        *   Mix and match to construct algorithms.
        *   Some combinations are more parallel-friendly than others.
        *   Monoids matter!

Parallel prefix computation:

*   Also called "parallel scan"
*   Scan variations:
    *   Exclusive vs inclusive:
        *   $[0, a_0,a_0+a_1, ...]$ and $a_0+a_1+\cdots$
        *   $[a_0,a_0+a_1, ...]$
    *   Prefix vs suffix
    *   We'll look at exclusive, prefix
*   Simple sequential algorithm (on arrays)
    *   *In C with mutation*
*   Simple sequential algorithm (on lists)
    *   *In Haskell without mutation*
*   Arrays, reasoning, and parallelism
    *   Array is a very common choice for parallel programming
    *   Constant-time random access,
        *   but not really.
    *   Down side: arrays are monolithic.
        *   Either an small element or a large array.
        *   We lose compositionality of reasoning & programming.
        *   Symptoms:
            *   Index fiddling
            *   Array slicing (perhaps implicit)
            *   Bounds checking/errors
    *   If not arrays, what?
*   Divide and conquer
    *   Many array algorithms require size $2^n$.
        *   Sub-problems are half-size ($2^{n-1}$).
        *   Where else do we see this pattern?
        *   Binary trees
            *   Specifically, perfect binary leaf trees
            *   Two variants:
                *   Top-down: pair of trees (common)
                *   Bottom-up: tree of pairs (uncommon, but useful)
    *   Non-$2^n$ arrays via non-perfect trees

Idea for presenting parallel scan:

*   Do top-down:
    *   Scan left
    *   Scan right
    *   Adjust right and sum
*   Aesthetic criticism: asymmetric--we tweak right and sum but not left.
*   Analysis:
    *   Depth: $\log n$
    *   Work: $n \log n$
*   Next consider wider branching: ternary, quaternary, ..., octary, etc.
    *   What changes?
    *   We have to adjust *all but the first* subtree.
    *   Note sequence of adjustments.
        Looks familiar?

## Prefix computations

Given a sequence of elements $[a_0,a_1,\ldots,a_{n-1}]$, compute sums of all sequence prefixes:

$$[0, a_0+a_1, ...,a_0+a_1+\cdots+a_{n-1}]$$


Variations:

*   Exclusive vs inclusive:
    *   $[0, a_0,a_0+a_1, ...]$ and $a_0+a_1+\cdots$
    *   $[a_0,a_0+a_1, ...]$
*   Prefix vs suffix
*   We'll look at exclusive, prefix

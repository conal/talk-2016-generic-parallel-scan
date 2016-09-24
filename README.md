# Generic parallel scan

This talk continues a theme from [*Generic FFT*], namely elegant, *generic* construction of efficient, parallel-friendly algorithms. By "generic" (also called "polytypic"), I mean structured via very simple and general data structure building blocks. Such developments produce not just a single algorithm, but an infinite family of algorithms, one for each data structure that can be assembled from the building blocks.

This time, we'll delve into (parallel) prefix computations---known as "scans" to functional programmers. We'll start with a simple and precise scan specification that leads directly to an algorithm performing quadratic work and hence quadratic time in a sequential implementation. An easy observation leads to a familiar sequential algorithm performing linear work but does not lend itself to parallel execution. A conventional divide-and-conquer idea yields $O(\log n)$ parallel time (given sufficient computational resources) and $O(n \log n)$ work. Playing with variations and poking at asymmetries, a beautiful generalization reveals itself. An inversion of representation then reduces work to $O(n)$ while retaining logarithmic time.

These two scan algorithms fall out "for free" from right- vs left-association of functor compositions, revealing a hidden unity in what had appeared to be very different well-known algorithms. These data types, which I call (generalized) "top-down" and "bottom-up" trees, are exactly the same as we saw give rise to the well-known FFT algorithms "decimation in time" and "decimation in frequency" (DIT & DIF). I'll also show a bushier tree type that offers a compromise between the merits of top-down and bottom-up scans. FFT for bushes appears to improve over both DIT & DIF on both axes.

As with generic FFT, the star of the show is functor composition. The generic approach is more successful for scan than for FFT, however, since it applies to all five basic building blocks and thus works for a much larger collection of data types, including products and sums.

The content of this talk largely overlaps with [one][*Understanding efficient parallel scan*] I gave at our meetup three years ago. The new version will streamline some of the discussion, modernize and simplify the data types used to match those in the FFT talk, and show additional parallel computations/circuits.


[*Generic FFT*]: https://github.com/conal/talk-2016-generic-fft "talk by Conal Elliott (2016)"

[*Understanding efficient parallel scan*]: https://github.com/conal/talk-2013-understanding-parallel-scan "talk by Conal Elliott (2013)"

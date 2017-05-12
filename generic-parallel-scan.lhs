%% -*- latex -*-

% Presentation
%\documentclass[aspectratio=1610]{beamer} % Macbook Pro screen 16:10
% \documentclass{beamer} % default aspect ratio 4:3
\documentclass[handout]{beamer}

% \setbeameroption{show notes} % un-comment to see the notes

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

\title{Generic parallel scan}
\date[October 2016]{October 5, 2016 (revised May 2017)}
% \date{\emph{[\today]}}

\setlength{\blanklineskip}{1.5ex}
\setlength\mathindent{4ex}

\begin{document}

\frame{\titlepage}

\framet{Some applications of parallel scan}{
From a longer list in \href{http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.128.6230}{\emph{Prefix
Sums and Their Applications}}:
\vspace{0.5ex}
\begin{itemize}\itemsep3ex
%% \item Lexical ordering
\item Adding multi-precision numbers
\item Polynomial evaluation
\item Solving recurrences
\item Sorting
\item Solving tridiagonal linear systems
%% \item Delete marked elements from an array
%% \item Dynamic processor allocation
\item Lexical analysis
\item Regular expression search
%% \item Some tree operations. For example, to find the depth of every vertex in a tree (see Chapter 3).
%% \item To label components in two dimensional images.
\end{itemize}
}

\framet{Prefix sum (left scan)}{
\vspace{5ex}
Given $a_1,\ldots,a_n$, compute

\vspace{3ex}
$$ {\Large b_k = \sum\limits_{1 \le i < k}{a_i}}\qquad\text{for~} k=1,\ldots, n+1$$

%% \vspace{3ex}
%% efficiently in work and depth (ideal parallel time).

\vspace{5ex}

Note that $a_k$ does \emph{not} influence $b_k$.
}

\framet{Linear left scan}{
\vspace{-2ex}
\wfig{4.5in}{circuits/lsums-lv8}

\vspace{-1ex} \pause
\emph{Work:} $O(n)$

%% \pause
\emph{Depth}: $O(n)$ (ideal parallel ``time'')

\vspace{2ex}
\pause
Linear \emph{dependency chain} thwarts parallelism (depth $<$ work).
}

\framet{Scan class}{
\vspace{10ex}
\begin{code}
class Functor f => LScan f where
  lscan :: Monoid a => f a -> f a :* a
\end{code}

\pause\vspace{10ex}
Specification (if |Traversable f|):
\begin{code}
lscan == swap . mapAccumL (\ acc a -> (acc <> a,acc)) mempty
\end{code}
%% where
%% \begin{code}
%% mapAccumL  :: Traversable t
%%            => (b -> a -> b :* c) -> b -> t a -> b :* t c
%% \end{code}
}

\framet{Generic building blocks}{
\vspace{2ex}
\begin{code}
data     V1           a                        -- void
newtype  U1           a = U1                   -- unit
newtype  Par1         a = Par1 a               -- singleton

data     (f  :+:  g)  a = L1 (f a) | R1 (g a)  -- sum
data     (f  :*:  g)  a = f a :*: g a          -- product
newtype  (g  :.:  f)  a = Comp1 (g (f a))      -- composition
\end{code}

\pause
\vspace{1ex}

Plan:

\begin{itemize}
\item Define parallel scan for each.
\item Use directly, \emph{or}
\item \hspace{2ex}automatically via (derived) encodings.
\end{itemize}
}

\framet{Easy instances}{

\begin{code}
instance LScan V1    where lscan = \ SPC case

instance LScan U1    where lscan U1 = (U1, mempty)

instance LScan Par1  where lscan (Par1 a) = (Par1 mempty, a)
\end{code}
%if False
\vspace{-6ex}
\begin{center}
\ccap{|U1|}{1.2in}{lsums-u}
\ccap{|Par1|}{1in}{lsums-i}
\end{center}
\vspace{-3ex}
%endif
\pause
\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = first L1  (lscan fa  )
  lscan (R1  ga  ) = first R1  (lscan ga  )
\end{code}
}

%format Vec = LVec

%% $a^{16} = a^5 \times a^{11}$
\framet{Product example: |Vec N5 :*: Vec N11|}{
\vspace{-2ex}
\wfig{2.3in}{circuits/lsums-lv5}
\vspace{-5ex}
\wfig{4.5in}{circuits/lsums-lv11}

\emph{Then what?}
}

\ccircuit{Combine?}{0}{lsums-lv5-lv11-unknown-no-hash}
\ccircuit{Combine?}{0}{lsums-lv5-lv11-unknown-no-hash-highlight}
\ccircuit{Right adjustment}{0}{lsums-lv5xlv11-highlight}

\framet{Products}{
\begin{textblock}{200}[1,0](350,5)
\begin{tcolorbox}
\wfig{2.5in}{circuits/lsums-lv5xlv11-highlight}
\end{tcolorbox}
\end{textblock}
\pause\vspace{23ex}
\begin{code}
instance (LScan f, LScan g) => LScan (f :*: g) where
  lscan (fa :*: ga) = (fa' :*: ((fx <> NOP) <#> ga'), fx <> gx)
   where
     (fa'  , fx)  = lscan fa
     (ga'  , gx)  = lscan ga
\end{code}
}

\framet{Vector type families}{
\vspace{3ex}
Right-associated:
\begin{code}
type family RVec_n where
  RVec Z      = U1
  RVec (S n)  = Par1 :*: RVec n
\end{code}

\pause\vspace{2ex}

Left-associated:
\begin{code}
type family LVec_n where
  LVec Z      = U1
  LVec (S n)  = LVec n :*: Par1
\end{code}

\pause\vspace{2ex}

Also convenient:
\begin{code}
type Pair = Par1 :*: Par1   -- or |RVec N2| or |LVec N2|
\end{code}
}

\circuit{|RVec N8| (unoptimized)}{-1}{lsums-rv8-no-hash-no-opt}{36}{8}
\circuit{|RVec N8| (optimized)}{-1}{lsums-rv8}{28}{7}

\circuit{|LVec N8| (unoptimized)}{0}{lsums-lv8-no-hash-no-opt}{16}{8}
\circuit{|LVec N8| (optimized)}{0}{lsums-lv8}{7}{7}

\circuit{|LVec N16| (optimized)}{0}{lsums-lv16}{15}{15}

\circuit{|LVec N5 :*: LVec N11|}{0}{lsums-lv5xlv11}{25}{11}
\circuit{$5+11$}{0}{lsums-lv5xlv11}{25}{11}
\circuit{$11+5$}{0}{lsums-lv11xlv5}{19}{11}
\circuit{$8+8$}{0}{lsums-p-lv8}{22}{8}
\circuit{$(5+5)+6$}{0}{lsums-lv5-5-6-l}{24}{6}
\circuit{$5+(5+6)$}{-1}{lsums-lv5-5-6-r}{30}{7}

\framet{Composition example: |LVec N3 :.: LVec N4|}{
\vspace{-3ex}
\wfig{2.5in}{circuits/lsums-lv4}
\vspace{-5ex}
\wfig{2.5in}{circuits/lsums-lv4}
\vspace{-5ex}
\wfig{2.5in}{circuits/lsums-lv4}
\vspace{-5ex}
\emph{Then what?}
}

\ccircuit{Combine?}{-1}{lsums-lv3olv4-unknown-no-hash}
\ccircuit{$(4+4)+4$}{0}{lsums-lv3olv4}
\ccircuit{$3 \times 4$}{0}{lsums-lv3olv4}
\ccircuit{|LVec N3 :.: LVec N4|}{0}{lsums-lv3olv4}
\ccircuit{|LVec N3 :.: LVec N4|}{0}{lsums-lv3olv4-highlight}

\ccircuit{$(((7+7)+7)+7)+7$}{-1.5}{lsums-lv5olv7}
%% \ccircuit{$5 \times 7$}{-1.5}{lsums-lv5olv7}
\ccircuit{|LVec N5 :.: LVec N7|}{-1.5}{lsums-lv5olv7}
\ccircuit{|LVec N5 :.: LVec N7|}{-1.5}{lsums-lv5olv7-highlight}

\framet{Composition}{
\begin{textblock}{200}[1,0](350,5)
\begin{tcolorbox}
\wfig{2.5in}{circuits/lsums-lv3olv4-highlight}
\end{tcolorbox}
\end{textblock}
\pause\vspace{24ex}
\begin{code}
instance (LScan g, LScan f, Zip g) =>  LScan (g :.: f) where
  lscan (Comp1 gfa) = (Comp1 (zipWith adjustl tots' gfa'), tot)
   where
     (gfa', tots)  = unzip (lscan <#> gfa)
     (tots',tot)   = lscan tots
     adjustl t     = fmap (t <> NOP)
\end{code}
}

\circuit{|Pair :.: LVec N8|}{0}{lsums-p-lv8}{22}{8}
\circuit{$2 \times 8$}{0}{lsums-p-lv8}{22}{8}
\circuit{$8 \times 2$}{0}{lsums-lv8-p}{22}{8}
\circuit{$4 \times 4$}{0}{lsums-lv4olv4}{24}{6}
\circuit{$4^2$}{0}{lsums-lv4olv4}{24}{6}

\framet{Functor exponentiation as type families}{
\vspace{-3ex}
\begin{center}
\Large $f^n = \overbrace{f \circ \cdots \circ f}^{n \text{~times}}$
\end{center}
\pause\vspace{1ex}
Right-associated/top-down:

\begin{code}
type family RPow h n where
  RPow h Z      = Par1
  RPow h (S n)  = h :.: RPow h n
\end{code}
{}

Left-associated/bottom-up:
\begin{code}
type family LPow h n where
  LPow h Z      = Par1
  LPow h (S n)  = LPow h n :.: h
\end{code}
}

\circuit{|RPow (LVec N4) N2|}{0}{lsums-lpow-4-2}{24}{6}
\circuit{$4^2$}{0}{lsums-lpow-4-2}{24}{6}

\circuit{$\overrightarrow{2^4} = 2 \times (2 \times (2 \times 2))$}{-1}{lsums-rb4}{32}{4}
\circuit{$\overleftarrow{2^4} = ((2 \times 2) \times 2) \times 2$}{-1}{lsums-lb4}{26}{6}

\circuit{$2^4 = (2^2)^2 = (2 \times 2) \times (2 \times 2)$}{-1}{lsums-bush2}{29}{5}

\framet{Bushes}{
\vspace{5ex}
\begin{code}
type family Bush n where
  Bush Z      = Pair
  Bush (S n)  = Bush n :.: Bush n
\end{code}
\vspace{3ex}\pause

Notes:
\begin{itemize}
\item
Composition-balanced counterpart to |LPow| and |RPow|.
\item
Variation of |Bush| type in \href{http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.184.8120}{\emph{Nested Datatypes}} by Bird \& Meertens.
\item
Size $2^{2^n}$, i.e., $2, 4, 16, 256, 65536, \ldots$.
\item
Easily generalizes beyond pairing and squaring.
\end{itemize}
}

%% \circuit{|Bush N2|}{-1}{lsums-bush2}{29}{5}

\circuit{|Bush N0|}{0}{lsums-bush0}{1}{1}
\circuit{|Bush' N0|}{0}{lsums-bush0}{1}{1}
\circuit{|Bush' N1|}{0}{lsums-bush1}{4}{2}
\circuit{|Bush' N2|}{0}{lsums-bush2}{29}{5}
\circuit{|Bush' N3|}{0}{lsums-bush3}{718}{10}

\circuit{$\overrightarrow{2^8}$}{-1}{lsums-rb8}{1024}{8}
\circuit{$\overleftarrow{2^8}$}{-1}{lsums-lb8}{502}{14}

%if False
\framet{Parallel, bottom-up, binary tree scan in CUDA C}{
\pause
\begin{minipage}[c]{0.7\textwidth}
\tiny
\begin{verbatim}
__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    // load input into shared memory
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];
    // build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai]; }
        offset *= 2; }
    // clear the last element
    if (thid == 0) { temp[n - 1] = 0; }
    // traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; } }
    __syncthreads();
    // write results to device memory
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1]; }
\end{verbatim}
\vspace{-6ex}
\href{http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html}{Source: Harris, Sengupta, and Owens in \emph{GPU Gems 3}, Chapter 39}
\normalsize
\end{minipage}
\hspace{-1in}
\begin{minipage}[c]{0.25\textwidth}
\pause
\begin{figure}
\wpicture{2in}{beaker-looks-left}

%%\pause
\hspace{0.75in}\emph{WAT}
\end{figure}
\end{minipage}
}
%endif

\framet{Generic parallel scan}{

\begin{itemize}\itemsep1ex \setlength{\parskip}{0.5ex}
\item Parallel scan: useful for many parallel algorithms.
%if True
\item Parallel programming without arrays:
\begin{itemize}\itemsep1ex \setlength{\parskip}{0.5ex}
\item Safety (no indexing errors).
\item Functor shape guides algorithm shape.
\end{itemize}
%else
\item Some convenient data structures:
\begin{itemize}\itemsep1ex
  \item Right \& left vectors
        %% . Depth $O(n)$; work $O(n)$ / $O(n^2)$.
  \item Top-down \& bottom-up trees
        %% . Depth $O(\log n)$; work $O(n \log n)$ / $O(n)$.
  \item Bushes
  \item No arrays!
\end{itemize}
%endif
\item Generic programming:
  \begin{itemize}\itemsep1ex
  \item Define per functor building block.
  \item Use directly, \emph{or}
  \item \hspace{2ex} automatically via (perhaps derived) encodings.
  \item Infinite variations, easily explored and guaranteed correct.
  \end{itemize}
%if False
\item Future work:
  \begin{itemize}\itemsep1ex
  \item Finish complexity analysis (bushes)
  \item Derive each instance from the |Traversable|-based specification.
  \item |Monoid| vs |Semigroup|, e.g., |Max| with |RPow|, |LPow|, |Bush|, and non-empty left- and right-vectors.
  \end{itemize}
%endif
\item Related talk: \href{https://github.com/conal/talk-2016-generic-fft}{Generic FFT}
\item Paper: \href{http://conal.net/papers/generic-parallel-functional/}{Generic parallel functional programming}
\end{itemize}
}

\framet{Extras}{

\begin{itemize} \itemsep3ex
\item \hyperlink{encodings}{Data encodings}
\item \hyperlink{packaging}{Convenient packaging}
\item \hyperlink{polynomial}{Application: polynomial evaluation}
\item \hyperlink{addition}{Application: parallel addition}
\end{itemize}

}

%format Type = "\ast"
\framet{Data encodings}{
\label{encodings}

From |GHC.Generics|:
\vspace{2ex}

\begin{code}
class Generic1 f where
  type Rep1 f :: Type -> Type
  from1  :: f a -> Rep1 f a
  to1    :: Rep1 f a -> f a
\end{code}

\vspace{5ex}

For regular algebraic data types, say ``|... NOP deriving Generic1|''.

}

\framet{Scan class}{
\begin{code}
class Functor f => LScan f where
  lscan :: Monoid a => f a -> f a :* a
\end{code}
\pause\vspace{-8ex}
\begin{code}
NOP
  default lscan  ::  (Generic1 f, LScan (Rep1 f))
                 =>  Monoid a => f a -> f a :* a
  lscan = first to1 . lscan . from1
\end{code}
}

\framet{Vector type families}{
\vspace{3ex}
Right-associated:
\begin{code}
type family RVec_n where
  RVec Z      = U1
  RVec (S n)  = Par1 :*: RVec n
\end{code}

\vspace{2ex}

Left-associated:
\begin{code}
type family LVec_n where
  LVec Z      = U1
  LVec (S n)  = LVec n :*: Par1
\end{code}

\vspace{0ex}
}

\framet{Vector GADTs}{
\begin{code}
data RVec NOP :: Nat -> STAR -> STAR SPC where
  ZVec  ::                   RVec Z      a
  (:<)  :: a -> RVec n a ->  RVec (S n)  a

instance Generic1 (RVec Z) where
  type Rep1 (RVec Z) = U1
  from1 ZVec = U1
  to1 U1 = ZVec

instance Generic1 (RVec (S n)) where
  type Rep1 (RVec (S n)) = Par1 :*: RVec n
  from1 (a :< as) = Par1 a :*: as
  to1 (Par1 a :*: as) = a :< as

instance                    LScan (RVec Z)
instance LScan (RVec n) =>  LScan (RVec (S n))
\end{code}

Plus |Functor|, |Applicative|, |Foldable|, |Traversable|, |Monoid|, |Key|, \ldots.
}

\framet{Functor exponentiation type families}{
\vspace{-3ex}
\begin{center}
\Large $f^n = \overbrace{f \circ \cdots \circ f}^{n \text{~times}}$
\end{center}
\vspace{0ex}
Right-associated/top-down:

\begin{code}
type family RPow h n where
  RPow h Z      = Par1
  RPow h (S n)  = h :.: RPow h n
\end{code}

Left-associated/bottom-up:
\begin{code}
type family LPow h n where
  LPow h Z      = Par1
  LPow h (S n)  = LPow h n :.: h
\end{code}

\ 
}

\framet{Functor exponentiation GADTs}{
\vspace{-3ex}
\begin{center}
\Large $f^n = \overbrace{f \circ \cdots \circ f}^{n \text{~times}}$
\end{center}

Right-associated/top-down:
\begin{code}
data RPow :: (STAR -> STAR) -> Nat -> STAR -> STAR where
  L :: a               -> RPow h Z      a
  B :: h (RPow h n a)  -> RPow h (S n)  a
\end{code}

Left-associated/bottom-up:
\begin{code}
data LPow :: (STAR -> STAR) -> Nat -> STAR -> STAR where
  L  :: a               -> LPow h Z      a
  B  :: LPow h n (h a)  -> LPow h (S n)  a
\end{code}

Plus |Generic1|, |Functor|, |Foldable|, |Traversable|, |Monoid|, |Key|, \ldots.

}

\framet{Some convenient packaging}{
\label{packaging}
\begin{code}
lscanAla  ::  forall n o f. (Newtype n, o ~ O n, LScan f, Monoid n)
          =>  f o -> f o :* o
lscanAla = (fmap unpack *** unpack) . lscan . fmap (pack @n)
NOP
lsums      = lscanAla @(Sum a)
lproducts  = lscanAla @(Product a)
lalls      = lscanAla @All
...
\end{code}

\pause Some simple uses:
\begin{code}
multiples  = lsums      . point
powers     = lproducts  . point
\end{code}
}

\circuit{|lproducts @(RBin N4)|}{-1}{lproducts-rb4}{32}{4}
%% \circuit{|point @(RBin N4)|}{0}{point-rb4}{0}{0}
\framet{|point @(RBin N4)|}{\vspace{1ex}\wfig{3.5in}{circuits/point-rb4}}
\circuit{|powers @(RBin N4)|}{-1}{powers-rb4-no-hash}{32}{4}
\circuit{|powers @(RBin N4)| --- with CSE}{-1}{powers-rb4}{15}{4}

\framet{Example: polynomial evaluation}{
\label{polynomial}

\begin{code}
evalPoly  ::  (LScan f, Foldable f, Zip f, Pointed f, Num a)
          =>  f a -> a -> a
evalPoly coeffs x = coeffs <.> fst (powers x)

NOP

(<.>) :: (Foldable f, Zip f, Num a) => f a -> f a -> a
u <.> v = sum (zipWith (*) u v)
\end{code}
}

\circuit{|(<.>) @(RBin N4)|}{-0.5}{dot-rb4}{16+15}{5}
\circuit{|evalPoly @(RBin N4)|}{0}{evalPoly-rb4}{29+15}{9}

\framet{Addition}{
\label{addition}
\pause
Generate and propagate carries:
\begin{code}
data PropGen = PropGen Bool Bool
\end{code}

\begin{code}
propGen :: Bool -> Bool -> PropGen
propGen a b = PropGen (a `xor` b) (a && b)  -- half adder
\end{code}

\pause\vspace{1ex}

\begin{code}
instance Monoid PropGen where
  mempty = PropGen True False
  PropGen pa ga `mappend` PropGen pb gb =
    PropGen (pa && pb) ((ga && pb) || gb)
\end{code}
}

\circuit{|scanAdd @(LVec' N8)|}{0}{scanAdd-lv8}{37}{15}
\circuit{|scanAdd @(RBin N3)|}{0}{scanAdd-rb3}{52}{8}
\circuit{|scanAdd @(LBin N3)|}{0}{scanAdd-lb3}{49}{10}

\circuit{|scanAdd @(RBin N4)|}{0}{scanAdd-rb4}{128}{10}
\circuit{|scanAdd @(LBin N4)|}{0}{scanAdd-lb4}{110}{14}
\circuit{|scanAdd @(Bush' N2)|}{0}{scanAdd-bush2}{119}{12}

\end{document}

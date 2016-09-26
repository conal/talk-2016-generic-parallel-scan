%% -*- latex -*-

%% %let long = True

% Presentation
%\documentclass[aspectratio=1610]{beamer} % Macbook Pro screen 16:10
\documentclass{beamer} % default aspect ratio 4:3
% \documentclass[handout]{beamer}

% \setbeameroption{show notes} % un-comment to see the notes

\usefonttheme{serif}
\usepackage{framed}

\usepackage{hyperref}
\usepackage{color}

\definecolor{linkColor}{rgb}{0,0.42,0.3}
\definecolor{partColor}{rgb}{0,0,0.8}

\hypersetup{colorlinks=true,urlcolor=linkColor}

\usepackage{graphicx}
\usepackage{color}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}

\usepackage{geometry}
% \usepackage[a4paper]{geometry}

%% \usepackage{wasysym}
\usepackage{mathabx}
\usepackage{setspace}
\usepackage{enumerate}
\usepackage{tikzsymbols}
\usepackage{fancybox}

\usepackage[absolute,overlay]{textpos}  % ,showboxes

\TPGrid{364}{273} %% roughly page size in points

\useinnertheme[shadow]{rounded}
% \useoutertheme{default}
\useoutertheme{shadow}
\useoutertheme{infolines}
% Suppress navigation arrows
\setbeamertemplate{navigation symbols}{}

\newcommand\sourced[1]{\href{#1}{\tiny (source)}}

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

\definecolor{statColor}{rgb}{0,0.5,0}

\newcommand{\stats}[2]{
{\small \textcolor{statColor}{work: #1, depth: #2}}}

\newcommand\ccircuit[3]{
\framet{#1}{
\vspace{#2ex}
\wfig{4.5in}{circuits/#3}
}}

\newcommand\circuit[5]{
\ccircuit{#1 \hfill \stats {#4}{#5}\hspace{2ex}}{#2}{#3}
}

\title{Generic parallel scan}
\author{\href{http://conal.net}{Conal Elliott}}
\institute{Target}
% \date{October 5, 2016}
\date{\emph{[\today]}}

\setlength{\itemsep}{2ex}
\setlength{\parskip}{1ex}
\setlength{\blanklineskip}{1.5ex}
\setlength\mathindent{4ex}
% \setstretch{1.2} % ??

\graphicspath{{Figures/}}

\definecolor{shadecolor}{rgb}{0.95,0.95,0.95}
\setlength{\fboxsep}{0.75ex}
\setlength{\fboxrule}{0.15pt}
\setlength{\shadowsize}{2pt}

%% \nc\cbox[1]{\raisebox{-0.5\height}{\fbox{#1}}}
\nc\cpic[2]{\fbox{\wpicture{#1}{circuits/#2}}}
\nc\ccap[3]{
\begin{minipage}[c]{0.48\textwidth}
\begin{center}
\cpic{#2}{#3}\par\vspace{0.5ex}#1\par
\end{center}
\end{minipage}
}

\begin{document}

\frame{\titlepage}

\framet{Prefix sum (left scan)}{
Given $a_1,\ldots,a_n$, compute

\vspace{3ex}
$$ {\Large b_k = \sum\limits_{1 \le i < k}{a_i}}\qquad\text{for~} k=1,\ldots, n+1$$
\vspace{3ex}

efficiently in work and depth (ideal parallel time).

\vspace{5ex}

Note that $a_k$ does \emph{not} influence $b_k$.
}

\framet{Some applications}{
From a longer list in \href{http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.128.6230}{\emph{Prefix
Sums and Their Applications}}:
\begin{itemize}\itemsep1.7ex
\item Lexical ordering
\item Adding multi-precision numbers
\item Polynomial evaluation
\item Solving recurrences
\item Radix sort
\item Quicksort
\item Solving tridiagonal linear systems
%% \item Delete marked elements from an array
%% \item Dynamic processor allocation
\item Lexical analysis.
\item Regular expression search
%% \item Some tree operations. For example, to find the depth of every vertex in a tree (see Chapter 3).
%% \item To label components in two dimensional images.
\end{itemize}
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
class LScan f where
  lscan :: Monoid a => f a -> f a :* a
\end{code}

\pause\vspace{10ex}
Specification (if |Traversable|):
\begin{code}
lscan == swap . mapAccumL (\ tot a -> (tot <> a,tot)) mempty
\end{code}
%% where
%% \begin{code}
%% mapAccumL  :: Traversable t
%%            => (b -> a -> b :* c) -> b -> t a -> b :* t c
%% \end{code}
}

\framet{Generic programming}{
\vspace{2ex}
\texttt{GHC.Generics}:
\begin{code}
data     V1           a                        -- lifted |Void|
newtype  U1           a = U1                   -- lifted |()|
newtype  Par1         a = Par1 a               -- singleton

data     (f  :+:  g)  a = L1 (f a) | R1 (g a)  -- lifted |Either|
data     (f  :*:  g)  a = f a :*: g a          -- lifted |(,)|
newtype  (g  :.:  f)  a = Comp1 (g (f a))      -- composition
\end{code}

\pause
\vspace{1ex}

Plan:

\begin{itemize}
\item Define parallel scan for each.
\item Use directly, \emph{or}
\item automatically via (derived) |Generic1| instances.
\end{itemize}
}

\framet{Scan class}{
\vspace{3ex}
\begin{code}
class LScan f where
  lscan :: Monoid a => f a -> And1 f a
  
  default lscan  ::  (Generic1 f, LScan (Rep1 f))
                 =>  Monoid a => f a -> And1 f a
  lscan = firstAnd1 to1 . lscan . from1
\end{code}

\pause\vspace{2ex}
% \hrule
\vspace{2ex}

\begin{code}
type And1 f = f :*: Par1

pattern (:>) :: f a -> a -> And1 f a
pattern fa :> a = fa :*: Par1 a

firstAnd1 :: (f a -> g a) -> And1 f a -> And1 g a
firstAnd1 q (fa :> a) = q fa :> a
\end{code}
}

\framet{Easy instances}{

\begin{code}
instance LScan V1    where lscan = \ SPC case

instance LScan U1    where lscan = (NOP :> mempty)

instance LScan Par1  where lscan (Par1 a) = Par1 mempty :> a
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
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
}

%% \circuit{Linear example}{0}{lsums-lv16}{15}{15}
%% \circuit{Products: $a^{m+n} = a^m \times a^n$}{0}{lsums-lv16}{15}{15}

%if True
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
\ccircuit{Right bump}{0}{lsums-lv5xlv11-highlight}
%% \ccircuit{Right bump}{0}{lsums-lv5xlv11}

%else
\framet{Divide and conquer? \hfill \stats {14}{7}\hspace{2ex}}{
\vspace{-2ex}
\wfig{4.5in}{circuits/lsums-lv8-wide}
\vspace{-5ex}
\wfig{4.5in}{circuits/lsums-lv8-wide}
\vspace{-4ex}
\emph{Then what?}
}

\circuit{Divide and conquer?}{0}{lsums-lv8-lv8-unknown-no-hash}{14+?}{7+?}
\circuit{Divide and conquer}{0}{lsums-p-lv8}{22}{8}
%endif

\framet{Products}{
\begin{code}
instance (LScan f, LScan g, Functor g) => LScan (f :*: g) where
  lscan (fa :*: ga) = (fa' :*: ga') :> gx
   where
     fa'  :>  fx  =                 lscan fa
     ga'  :>  gx  = adjustl fx <#>  lscan ga

SPC

adjustl :: (Monoid a, Functor g) => a -> g a -> g a
adjustl a as = (a <> NOP) <#> as
\end{code}
}

\framet{Vector GADT}{\pause
\begin{code}
data RVec NOP :: Nat -> * -> * SPC where
  ZVec  :: RVec Z a 
  (:<)  :: a -> RVec n a -> RVec (S n) a
\end{code}
\pause\vspace{-4ex}
\begin{code}
instance                    LScan (RVec Z)
instance LScan (RVec n) =>  LScan (RVec (S n))
\end{code}
\pause\vspace{-4ex}
\begin{code}
instance Generic1 (RVec Z) where
  type Rep1 (RVec Z) = U1
  from1 ZVec = U1
  to1 U1 = ZVec

instance Generic1 (RVec (S n)) where
  type Rep1 (RVec (S n)) = Par1 :*: RVec n
  from1 (a :< as) = Par1 a :*: as
  to1 (Par1 a :*: as) = a :< as

\end{code}

Plus |Functor|, |Applicative|, |Foldable|, |Traversable|, |Monoid|, |Key|, \ldots.
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
%% \ccircuit{|LVec N5 :.: LVec N7|}{-1.5}{lsums-lv5olv7}
\ccircuit{|LVec N5 :.: LVec N7|}{-1.5}{lsums-lv5olv7-highlight}

\framet{Composition}{
\vspace{3ex}
\begin{code}
instance          (LScan g, LScan f, Functor f, Zip g)
  SPACE SPACE =>  LScan (g :.: f) where
  lscan (Comp1 gfa) = Comp1 (zipWith adjustl tots' gfa') :> tot
   where
     (gfa', tots)  = unzipAnd1 (lscan <#> gfa)
     tots' :> tot  = lscan tots
\end{code} % 

\vspace{3ex}

\begin{code}
unzipAnd1 :: forall g f a. Functor g => g (And1 f a) -> g (f a) :* g a
unzipAnd1 = unzip . fmap (\ (as :> a) -> (as,a))

unzip :: Functor f => f (a, b) -> (f a, f b)
unzip ps = (fst <#> ps, snd <#> ps)
\end{code}
}

\circuit{|Pair :.: LVec N8|}{0}{lsums-p-lv8}{22}{8}
\circuit{$2 \times 8$}{0}{lsums-p-lv8}{22}{8}
\circuit{$8 \times 2$}{0}{lsums-lv8-p}{22}{8}
\circuit{$4 \times 4$}{0}{lsums-lv4olv4}{24}{6}
\circuit{$4^2$}{0}{lsums-lv4olv4}{24}{6}

\framet{Exponentiation as GADTs}{

Top-down, depth-typed, perfect, leaf trees
\begin{code}
data RPow :: (* -> *) -> Nat -> * -> * NOP where
  L :: a               -> RPow h Z      a
  B :: h (RPow h n a)  -> RPow h (S n)  a
\end{code}

\pause\vspace{3ex}

Bottom-up, depth-typed, perfect, leaf trees:
\begin{code}
data LPow :: (* -> *) -> Nat -> * -> * NOP where
  L  :: a               -> LPow h Z      a
  B  :: LPow h n (h a)  -> LPow h (S n)  a
\end{code}

Plus |Generic1|, |Functor|, |Foldable|, |Traversable|, |Monoid|, |Key|, \ldots.

}

\framet{Exponentiation as type families}{

Right-associated/top-down:

> type family RPow h n where
>   RPow h Z      = Par1
>   RPow h (S n)  = h :.: RPow h n

{}

Left-associated/bottom-up:

> type family LPow h n where
>   LPow h Z      = Par1
>   LPow h (S n)  = LPow h n :.: h

}

\circuit{|LPow (LVec N4) N2|}{0}{lsums-lpow-4-2}{24}{6}
\circuit{$4^2$}{0}{lsums-lpow-4-2}{24}{6}

%% \circuit{|LPow Pair N4|}{-1}{lsums-lb4}{26}{6}
\circuit{$\overleftarrow{2^4} = ((2 \times 2) \times 2) \times 2$}{-1}{lsums-lb4}{26}{6}

\circuit{$\overrightarrow{2^4} = 2 \times (2 \times (2 \times 2))$}{-1}{lsums-rb4}{32}{4}
%% \circuit{|RPow Pair N4|}{-1}{lsums-rb4}{32}{4}

\circuit{$2^4 = (2^2)^2 = (2 \times 2) \times (2 \times 2)$}{-1}{lsums-bush2}{29}{5}

\framet{Bushes}{
\vspace{5ex}

> type family Bush n where
>   Bush Z      = Pair
>   Bush (S n)  = Bush n :.: Bush n

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
\circuit{$2^{2^0}$}{0}{lsums-bush0}{1}{1}
%\circuit{|Bush N1|}{0}{lsums-bush1}{4}{2}
\circuit{$2^{2^1}$}{0}{lsums-bush1}{4}{2}
\circuit{$2^{2^2}$}{0}{lsums-bush2}{29}{5}
\circuit{$2^{2^3}$}{0}{lsums-bush3}{718}{10}

\framet{Parallel, bottom-up, binary tree scan in CUDA C}{
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

\framet{Some convenient packaging}{

\begin{code}
lscanAla  ::  (Newtype n, o ~ O n, LScan f, Monoid n)
          =>  (o -> n) -> f o -> And1 f o
lscanAla = flip underF lscan

lsums      = lscanAla Sum
lproducts  = lscanAla Product
lalls      = lscanAla All
...
\end{code}
\pause Some simple uses:
\begin{code}
multiples  = lsums      . point
powers     = lproducts  . point

multiplicationTable  = multiples  <$> multiples  1
fiddleFactors        = powers     <$> powers     omega
\end{code}
}

\circuit{|lproducts @(RPow Pair N4)|}{-1}{lproducts-rb4}{32}{4}
\framet{|point @(RPow Pair N4)|}{\wfig{3in}{circuits/point-rb4}}
%% \ccircuit{|point @(RPow Pair N4)|}{0}{point-rb4}
\circuit{|powers @(RPow Pair N4)|}{-1}{powers-rb4-no-hash}{32}{4}
\circuit{|powers @(RPow Pair N4)| --- with CSE}{-1}{powers-rb4}{15}{4}

\framet{Example: polynomial evaluation}{

\begin{code}
evalPoly  ::  (LScan f, Foldable f, Zip f, Pointed f, Num a)
          =>  And1 f a -> a -> a
evalPoly coeffs x = coeffs <.> powers x

NOP

(<.>) :: (Foldable f, Zip f, Num a) => f a -> f a -> a
u <.> v = sum (zipWith (*) u v)
\end{code}
}

\circuit{|(<.>) @(And1 (RPow Pair N4))|}{0}{dot-rb4-1}{17+16}{6}
\circuit{|evalPoly @(RPow Pair N4)|}{0}{evalPoly-rb4}{31+16}{10}

\end{document}

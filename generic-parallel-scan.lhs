%% -*- latex -*-

%% %let long = True

% Presentation
%\documentclass[aspectratio=1610]{beamer} % Macbook Pro screen 16:10
% \documentclass{beamer} % default aspect ratio 4:3
\documentclass[handout]{beamer}

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

\nc\bboxed[1]{\boxed{\rule[-0.9ex]{0pt}{2.8ex}#1}}
\nc\vox[1]{\bboxed{#1}}
\nc\tvox[2]{\vox{#1}\vox{#2}}

\nc\lscan{\Varid{lscan}}

\nc\trans[1]{\\[1.3ex] #1 \\[0.75ex]}
\nc\ptrans[1]{\pause\trans{#1}}
\nc\ptransp[1]{\ptrans{#1}\pause}

\nc\pitem{\pause \item}

%%%%

% \setbeameroption{show notes} % un-comment to see the notes

\graphicspath{{Figures/}}

\definecolor{shadecolor}{rgb}{0.95,0.95,0.95}

\begin{document}

\frame{\titlepage}

\framet{Prefix sum (left scan)}{
\wfig{4.5in}{circuits/lsumsp-lv8}

\vspace{1ex} \pause
\emph{Work:} $O(n)$

\pause
\emph{Depth}: $O(n)$ (ideal parallel ``time'')

\vspace{2ex}
\pause
Linear \emph{dependency chain} thwarts parallelism (depth $<$ work).
}

\definecolor{statColor}{rgb}{0,0.5,0}

\newcommand{\stats}[2]{
{\small \textcolor{statColor}{work: #1, depth: #2}}}

\newcommand\circuit[5]{
\framet{#1 \hfill \stats {#4}{#5}\hspace{2ex}}{
\vspace{#2ex}
\wfig{4.5in}{circuits/#3}
}}

\circuit{Prefix sum (left scan)}{0}{lsumsp-lv16}{15}{15}

%% \circuit{Divide and conquer?}{3}{lsumsp-lv8-and-lv8}{14}{7}

\circuit{Divide and conquer?}{3}{lsumsp-lv8-lv8-unknown}{14+?}{7+?}

%% \framet{Divide and conquer? \hfill \stats {14}{7}\hspace{2ex}}{
%% \vspace{0ex}
%% \wfig{4.5in}{circuits/lsumsp-lv8-wide}
%% \wfig{4.5in}{circuits/lsumsp-lv8-wide}
%% \vspace{-2ex}
%% \emph{Then what?}
%% }

\circuit{Divide and conquer}{0}{lsumsp-p-lv8}{22}{8}

\circuit{$5+11$}{1}{lsumsp-lv5xlv11}{25}{11}
\circuit{$(5+5)+6$}{0}{lsumsp-lv5-5-6-l}{24}{6}
\circuit{$5+(5+6)$}{-1}{lsumsp-lv5-5-6-r}{30}{7}
\circuit{$8+8$}{0}{lsumsp-p-lv8}{22}{8}

\circuit{$2 \times 8$}{0}{lsumsp-p-lv8}{22}{8}
\circuit{$8 \times 2$}{0}{lsumsp-lv8-p}{22}{8}
\circuit{$4 \times 4$}{0}{lsumsp-lv4olv4}{24}{6}

\circuit{$4^2$}{0}{lsumsp-lv4olv4}{24}{6}

\circuit{$2^4 = ((2 \times 2) \times 2) \times 2$}{-1}{lsumsp-lb4}{26}{6}

\circuit{$2^4 = 2 \times (2 \times (2 \times 2))$}{-1}{lsumsp-rb4}{32}{4}

\circuit{$2^4 = (2^2)^2 = (2 \times 2) \times (2 \times 2)$}{-1}{lsumsp-bush2}{29}{5}

\framet{Generic programming}{
%\pause

\texttt{GHC.Generics}:
\begin{code}
data     V1         a                        -- lifted |Void|
newtype  U1         a = U1                   -- lifted |()|
newtype  Par1       a = Par1 a               -- singleton

data     (f :+: g)  a = L1 (f a) | R1 (g a)  -- lifted |Either|
data     (f :*: g)  a = f a :*: g a          -- lifted |(,)|
newtype  (g :.: f)  a = Comp1 (g (f a))      -- composition
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

\framet{Parallel scan as a class}{
\begin{code}
class LScan f where
  lscan :: Monoid a => f a -> f a
\end{code}
 
\pause
\vspace{-3ex}
\begin{code}
  SPACE  default lscan  ::  (Generic1 f, LScan (Rep1 f))
                        =>  Monoid a => f a -> f a
         lscan = to1 . lscan . from1
\end{code}
}

\framet{Easy instances}{

\begin{code}
instance LScan V1    where lscan = \ SPC case

instance LScan U1    where lscan = id

instance LScan Par1  where lscan = id
\end{code}

\pause
\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
}

\framet{Products}{
\vspace{3ex}
\begin{code}
instance (LScan f, LScan g, Functor g) => LScan (f :*: g) where
  lscan (fa :*: ga) = fa' :*: ga'
   where
     fa' =                                lscan fa
     ga' = (last' fa' `mappend` NOP) <#>  lscan ga

last' :: f a -> a
last' = ...
\end{code}

\pause\vspace{3ex}
\emph{Oops.}
Not all data structures have a last element.
}

\framet{Starting with zero}{
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

\framet{Easy instances, again}{

\begin{code}
instance LScan V1    where lscan = \ SPC case

instance LScan U1    where lscan = (NOP :> mempty)

instance LScan Par1  where lscan (Par1 a) = Par1 mempty :> a
\end{code}

\pause
\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
}

\framet{Products}{
\begin{code}
instance (LScan f, LScan g, Functor g) => LScan (f :*: g) where
  lscan (fa :*: ga) = (fa' :*: ga') :> gx
   where
     fa'  :>  fx  =                 lscan fa
     ga'  :>  gx  = adjustl fx <#>  lscan ga

SPC

adjustl :: (Monoid a, Functor t) => a -> t a -> t a
adjustl a as = (a <> NOP) <#> as
\end{code}
}

%% \nc\cbox[1]{\raisebox{-0.5\height}{\fbox{#1}}}
\nc\cpic[2]{\fbox{\wpicture{#1}{circuits/#2}}}
\nc\ccap[3]{
\begin{minipage}[c]{0.48\textwidth}
\begin{center}
\cpic{#2}{#3}\par\vspace{0.5ex}#1\par
\end{center}
\end{minipage}
}

\framet{Some simple scans}{
\setlength{\fboxsep}{0.75ex}
\setlength{\fboxrule}{0.15pt}
\setlength{\shadowsize}{2pt}
\vspace{-1ex}
\begin{center}
\ccap{|U1|}{1.2in}{lsums-u}
\ccap{|Par1|}{1in}{lsums-i}
\end{center}
\begin{center}
\ccap{|Par1 :*: U1|}{1.4in}{lsums-1-0-no-hash-no-opt}
\ccap{|Par1 :*: U1| (optimized)}{1in}{lsums-1-0}
\end{center}
\begin{center}
\ccap{|Par1 :*: (Par1 :*: U1)|}{1.5in}{lsums-1-1-0-no-hash-no-opt}
\ccap{|Par1 :*: (Par1 :*: U1)| (optimized)}{1.3in}{lsums-1-1-0}
\end{center}

}

\framet{|Par1 :*: (Par1 :*: (Par1 :*: (Par1 :*: U1)))| (unoptimized)}{
\vspace{0ex}
\wfig{4.5in}{circuits/lsums-1-1-1-1-0-r-no-hash-no-opt}
}
\circuit{$1+(1+(1+(1+0)))$ (unoptimized)}{0}{lsums-1-1-1-1-0-r-no-hash-no-opt}{10}{4}
\circuit{$1+(1+(1+(1+0)))$ (optimized)}{0}{lsums-1-1-1-1-0-r}{6}{3}

\circuit{$(((0+1)+1)+1)+1$ (unoptimized)}{1}{lsums-0-1-1-1-1-l-no-hash-no-opt}{8}{4}
\circuit{$(((0+1)+1)+1)+1$ (optimized)}{1}{lsums-0-1-1-1-1-l}{3}{3}

\framet{Vector as GADT}{
\begin{code}
data Vec NOP :: Nat -> * -> * SPC where
  ZVec  :: Vec Z a 
  (:<)  :: a -> Vec n a -> Vec (S n) a

instance                   LScan (Vec Z)
instance LScan (Vec n) =>  LScan (Vec (S n))
\end{code}
\pause\vspace{-4ex}
\begin{code}
instance Generic1 (Vec Z) where
  type Rep1 (Vec Z) = U1
  from1 ZVec = U1
  to1 U1 = ZVec

instance Generic1 (Vec (S n)) where
  type Rep1 (Vec (S n)) = Par1 :*: Vec n
  from1 (a :< as) = Par1 a :*: as
  to1 (Par1 a :*: as) = a :< as

\end{code}

Plus |Functor|, |Applicative|, |Foldable|, |Traversable|, |Monoid|, |Key|, \ldots.
}

\framet{Vector as type family}{
\begin{code}
type family Vec_n where
  Vec Z      = U1
  Vec (S n)  = Par1 :*: Vec n
\end{code}
}

\circuit{|Vec N8| (unoptimized)}{-1}{lsums-rv8-no-hash-no-opt}{36}{8}
\circuit{|Vec N8| (optimized)}{-1}{lsums-rv8}{28}{7}

\framet{Vector type families}{

Right-associated:
\begin{code}
type family RVec_n where
  RVec Z      = U1
  RVec (S n)  = Par1 :*: RVec n
\end{code}

\pause\vspace{3ex}

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

\circuit{|LVec N8| (unoptimized)}{1}{lsums-lv8-no-hash-no-opt}{16}{8}
\circuit{|LVec N8| (optimized)}{-1}{lsums-lv8}{7}{7}

\circuit{$8$}{0}{lsums-lv8}{7}{7}
\circuit{$16$}{0}{lsums-lv16}{15}{15}

\circuit{$(5+11)$ (unoptimized)}{1}{lsums-lv5xlv11-no-hash-no-opt}{25}{11}
\circuit{$(5+11)$ (optimized)}{1}{lsums-lv5xlv11}{25}{11}

\circuit{$(5+5)+6$}{0}{lsums-lv5-5-6-l}{24}{6}
\circuit{$5+(5+6)$}{-1}{lsums-lv5-5-6-r}{30}{7}
\circuit{$8+8$}{0}{lsums-p-lv8}{22}{8}
\circuit{$2 \times 8$}{0}{lsums-p-lv8}{22}{8}

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

\circuit{|LPow Pair N4|}{-1}{lsums-lb4}{26}{6}
\circuit{$\overleftarrow{2^4} = ((2 \times 2) \times 2) \times 2$}{-1}{lsums-lb4}{26}{6}

\circuit{|RPow Pair N4|}{-1}{lsums-rb4}{32}{4}
\circuit{$\overrightarrow{2^4} = 2 \times (2 \times (2 \times 2))$}{-1}{lsums-rb4}{32}{4}

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

\circuit{|Bush N1|}{0}{lsums-bush1}{4}{2}
\circuit{$2^{2^1}$}{0}{lsums-bush1}{4}{2}
\circuit{$2^{2^2}$}{0}{lsums-bush2}{29}{5}
\circuit{$2^{2^3}$}{0}{lsums-bush3}{718}{10}


\end{document}

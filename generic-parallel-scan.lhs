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

\circuit{Divide and conquer?}{3}{lsumsp-lv8-and-lv8}{14}{7}

\circuit{Divide and conquer \emph{with correction}}{0}{lsumsp-p-lv8}{22}{8}

\circuit{$5+11$}{3}{lsumsp-lv5xlv11}{25}{11}

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
data     V1         a                        -- lifted Void
newtype  K1 i c     a = K1 c                 -- constant
newtype  Par1       a = Par1 a               -- identity
data     (f :+: g)  a = L1 (f a) | R1 (g a)  -- lifted Either
data     (f :*: g)  a = f a :*: g a          -- lifted (,)
newtype  (f :.: g)  a = Comp1 (f (g a))      -- composition
\end{code}

\pause
\

Define parallel scan for each.

Applied automatically to data types with |Generic1| instances.

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
instance LScan V1 where lscan = \ SPC case
\end{code}

\pause
\begin{code}
instance LScan U1        where lscan = id
instance LScan (K1 i c)  where lscan = id

instance LScan Par1      where lscan = id
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
     ga' = (last' fa' `mappend` NOP) <$>  lscan ga

last' :: f a -> a
last' = ...
\end{code}

\pause\vspace{3ex}
\textcolor{red}{Oops.}
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

\pause\vspace{4ex}
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
instance LScan V1 where lscan = \ SPC case
\end{code}

\pause
\begin{code}
instance LScan U1        where lscan = id
instance LScan (K1 i c)  where lscan = id

instance LScan Par1      where lscan = id
\end{code}

\pause
\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
}

\end{document}

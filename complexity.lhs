%% -*- latex -*-

%let analysis=True

% Presentation
%\documentclass[aspectratio=1610]{beamer} % Macbook Pro screen 16:10
\documentclass{beamer} % default aspect ratio 4:3
% \documentclass[handout]{beamer}

% \setbeameroption{show notes} % un-comment to see the notes

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

\title{Some complexity analyses}
\subtitle{for generic parallel algorithms}
\date{October 5, 2016}
% \date{\emph{[\today]}}

\setlength{\blanklineskip}{1.5ex}
\setlength\mathindent{4ex}

\begin{document}

\frame{\titlepage}

\partframe{Scan}

\framet{Sums}{

\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
Analysis:
\begin{code}
Work   (f :+: g)  = max (Work   f) (Work   g)

Depth  (f :+: g)  = max (Depth  f) (Depth  g)
\end{code}
}

\framet{Products}{
\begin{textblock}{175}[1,0](350,8)
\begin{tcolorbox}
\wfig{2.2in}{circuits/lsums-lv5xlv11-highlight}
\end{tcolorbox}
\end{textblock}
\pause\vspace{17ex}
%if False
\begin{code}
instance (LScan f, LScan g) => LScan (f :*: g) where
  lscan (fa :*: ga) = (fa' :*: ((fx <> NOP) <#> ga'), fx <> gx)
   where
     (fa'  , fx)  = lscan fa
     (ga'  , gx)  = lscan ga
\end{code}
%endif
Analysis:
\begin{code}
Work   (f :*: g)  = Work f + Work g + Size g + 1

Depth  (f :*: g)  = max (Depth f) (Depth g) + 1
\end{code}
}

\framet{Composition}{
\begin{textblock}{175}[1,0](350,8)
\begin{tcolorbox}
\wfig{2.2in}{circuits/lsums-lv3olv4-highlight}
\end{tcolorbox}
\end{textblock}
\pause\vspace{19ex}
%if False
\begin{code}
instance (LScan g, LScan f, Zip g) =>  LScan (g :.: f) where
  lscan (Comp1 gfa) = (Comp1 (zipWith adjustl tots' gfa'), tot)
   where
     (gfa', tots)  = unzip (lscan <#> gfa)
     (tots',tot)   = lscan tots
     adjustl t     = fmap (t <>)
\end{code}
%endif
Analysis:
\begin{code}
Work   (g:.: f) = Size g * Work f + Work g + Size g * Size f

Depth  (g:.: f) = Depth(f) + Depth(g)
\end{code}
}

\framet{Right-associated vectors}{
\begin{textblock}{157}[1,0](350,5)
\setlength\mathindent{0.5ex}
\begin{tcolorbox}
\vspace{-1ex}
\begin{code}
type family RVec_n where
  RVec Z      = U1
  RVec (S n)  = Par1 :*: RVec n
\end{code}
\vspace{-3.5ex}
\end{tcolorbox}
\end{textblock}
\vspace{6ex}
\begin{code}
Work (RVec 0)      = 0
Work (RVec (S n))  = Work Par1 + Work (RVec n) + Size (RVec n) + 1
                   = Work (RVec n) + O(n)
                   ==>
Work (RVec n)      = O(pow n 2)
\end{code}
\vspace{1ex}
\begin{code}
Depth (RVec 0)      = 0
Depth (RVec (S n))  = max 0 Depth (RVec n) + 1
                    ==>
Depth (RVec n)      = O(n)
\end{code}
}

\framet{Left-associated vectors}{
\begin{textblock}{157}[1,0](350,5)
\setlength\mathindent{0.5ex}
\begin{tcolorbox}
\vspace{-1ex}
\begin{code}
type family LVec_n where
  LVec Z      = U1
  LVec (S n)  = LVec n :*: Par1
\end{code}
\vspace{-3.5ex}
\end{tcolorbox}
\end{textblock}
\vspace{6ex}
\begin{code}
Work (LVec 0)      = 0
Work (LVec (S n))  = Work (LVec n) + Work Par1 + Size Par1 + 1
                   = Work(LVec n) + O(1)
                   ==>
Work (LVec n)      = O(n)
\end{code}
\vspace{1ex}
\begin{code}
Depth (LVec 0)      = 0
Depth (LVec (S n))  = max (Depth (LVec n)) 0 + 1
                    ==>
Depth (LVec n)      = O(n)
\end{code}
}

\newcommand{\upperRPow}{
\begin{textblock}{180}[1,0](350,5)
\setlength\mathindent{0ex}
\begin{tcolorbox}
\vspace{-1ex}
\begin{code}
type family RPow h n where
  RPow h Z      = Par1
  RPow h (S n)  = h :.: RPow h n
\end{code}
\vspace{-3.5ex}
\end{tcolorbox}
\end{textblock}
}

\framet{Right-associated trees}{\upperRPow
\vspace{6ex}
\small \setlength\mathindent{0.5ex}
\begin{code}
Work (RPow h 0)      = 0
Work (RPow h (S n))  = Size h * Work (RPow h n) + Work h + Size (RPow h (S n))
                     = Size h * Work (RPow h n) + O (Size (RPow h (S n)))
                     ==>
Work (RPow h n)      = O (Size (RPow h n) * log (Size (RPow h n)))
\end{code}
\vspace{1ex}
\begin{code}
Depth (RPow h 0)      = 0
Depth (RPow h (S n))  = Depth h + Depth (RPow h n)
                      ==>
Depth (RPow h n)      = O (n)
                      = O (log (Size (RPow h n)))
\end{code}
}

\newcommand{\upperLPow}{
\begin{textblock}{180}[1,0](350,5)
\setlength\mathindent{0ex}
\begin{tcolorbox}
\vspace{-1ex}
\begin{code}
type family LPow h n where
  LPow h Z      = Par1
  LPow h (S n)  = LPow h n :.: h
\end{code}
\vspace{-3.5ex}
\end{tcolorbox}
\end{textblock}
}

\framet{Left-associated trees}{\upperLPow
\vspace{6ex}
\small \setlength\mathindent{0.5ex}
\begin{code}
Work (LPow h 0)      = 0
Work (LPow h (S n))  = Size (LPow h n) * Work h + Work (LPow h n) + Size (LPow h (S n))
                     = Work (LPow h n) + O (Size (LPow h (S n)))
                     ==>
Work (LPow h n)      = O (Size (LPow h n))
\end{code}
\vspace{1ex}
\begin{code}
Depth (LPow h 0)      = 0
Depth (LPow h (S n))  = Depth (LPow h n) + Depth h
                      ==>
Depth (LPow h n)      = O (n)
                      = O (log (Size (LPow h n)))
\end{code}
}

\framet{Bushes}{
\begin{textblock}{180}[1,0](350,5)
\setlength\mathindent{0ex}
\begin{tcolorbox}
\vspace{-1ex}
\begin{code}
type family Bush n where
  Bush Z      = Pair
  Bush (S n)  = Bush n :.: Bush n
\end{code}
\vspace{-3.5ex}
\end{tcolorbox}
\end{textblock}
\vspace{6ex}
\setlength\mathindent{0.25ex}
\begin{code}
Work (Bush 0)      = O (1)
Work (Bush (S n))  = Size (Bush n) * Work (Bush n) + Work (Bush n) + Size (Bush (S n))
                   ==>
Work (Bush n)      = ??
\end{code}
\vspace{1ex}
\begin{code}
Depth (Bush 0)      = 0
Depth (Bush (S n))  = Depth (Bush n) + Depth (Bush n)
                    = 2 * Depth (Bush n)
                    ==>
Depth (Bush n)      = pow 2 n
                    = log (Size (Bush n))
\end{code}
}

\partframe{FFT}

\framet{Composition}{
\wfig{4in}{cooley-tukey-general}
\begin{center}
\vspace{-5ex}
\sourced{https://en.wikipedia.org/wiki/Cooley\%E2\%80\%93Tukey_FFT_algorithm\#General_factorizations}
\end{center}
}

\framet{Composition}{
\begin{textblock}{180}[1,0](353,7)
\begin{tcolorbox}
\wpicture{2.2in}{cooley-tukey-general}
\end{tcolorbox}
\end{textblock}
\vspace{18ex}
\begin{code}
Work   (g :.: f)  = Size f * Work g + Size g * Size f + Size g * Work f

Depth  (g :.: f)  = Depth g + 1 + Depth f
\end{code}
}


\framet{Right-associated trees}{\upperRPow
\vspace{8ex}
\begin{code}
Work (RPow h Z)      = 0
Work (RPow h (S n))  = Work (h :.: RPow h n)
    =  Size (RPow h n) * Work h + Size (RPow h (S n)) + Size h * Work (RPow h n)
    =  O (Size (RPow h n)) + Size (RPow h (S n)) + O (Work (RPow h n))
    =  O (Work (RPow h n)) + O (Size (RPow h n))
                     ==>
Work (RPow h n)      = O (Size (RPow h n) * log (Size (RPow h n)))
\end{code}
}

\framet{Left-associated trees}{\upperLPow
\vspace{8ex}
\begin{code}
Work (LPow h Z)      = 0
Work (LPow h (S n))  = Work (LPow h n :.: h)
    =  Size h * Work (LPow h n) + Size (LPow h (S n)) + Size (LPow h n) * Work h
    =  O (Work (LPow h n)) + Size (LPow h (S n)) + O (Size (LPow h n))
    =  O (Work (LPow h n)) + O (Size (LPow h n))
                     ==>
Work (LPow h n)      = O (Size (LPow h n) * log (Size (LPow h n)))
\end{code}
}

\end{document}



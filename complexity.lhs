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

\DeclareMathOperator{\ParOne}{Par_1}
\DeclareMathOperator{\LVec}{LVec}
\DeclareMathOperator{\RVec}{RVec}
\DeclareMathOperator{\LPow}{LPow}
\DeclareMathOperator{\RPow}{RPow}

\begin{document}

\frame{\titlepage}

\partframe{Scan}

\framet{Sums}{

\begin{code}
instance (LScan f, LScan g) => LScan (f :+: g) where
  lscan (L1  fa  ) = L1  (lscan fa  )
  lscan (R1  ga  ) = R1  (lscan ga  )
\end{code}
%if analysis
Analysis:
\begin{align*}
\W (f \pmb{+} g) &= \max (\W f, \W g)\\
\D (f \pmb{+} g) &= \max (\D f, \D g)
\end{align*}
%endif
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
%if analysis
Analysis:
\begin{align*}
\W (f \pmb{\times} g) & = \W(f) + \W(g) + \Size g + 1 \\
\\
\D (f \pmb{\times} g) &= \max (\D(f), \D(g)) + 1
\end{align*}
%endif
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
%if analysis
Analysis:
\begin{align*}
\W (g \pmb{\circ} f) &= \Size g \times \W(f) + \W(g) + \Size g \times \Size f\\
\\
\D (g \pmb{\circ} f) &= \D(f) + \D(g)
\end{align*}
%endif
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
\begin{align*}
\W (\RVec_0) &= 0 \\
\W (\RVec_{1+n}) &= \W(\ParOne) + \W(\RVec_n) + \Size{\RVec_n} + 1 \\
                &= \W(\RVec_n) + O(n) \\
 &\therefore\\
\W (\RVec_n) &= O(n^2) \\
\end{align*}
\hrule
\begin{align*}
\D (\RVec_0) &= 0 \\
\D (\RVec_{1+n}) &= \max (0,\D(\RVec_n)) + 1\\
 &\therefore\\
\D (\RVec_n) &= O(n) \\
\end{align*}
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
\begin{align*}
\W (\LVec_0) &= 0 \\
\W (\LVec_{n+1}) &= \W(\LVec_n) + \W(\ParOne) + \Size{\ParOne} + 1 \\
                &= \W(\LVec_n) + O(1) \\
 &\therefore\\
\W (\LVec_n) &= O(n) \\
\end{align*}
\hrule
\begin{align*}
\D (\LVec_0) &= 0 \\
\D (\LVec_{n+1}) &= \max (\D(\LVec_n), 0) + 1\\
 &\therefore\\
\D (\LVec_n) &= O(n) \\
\end{align*}
}

\framet{Right-associated trees}{
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
\vspace{6ex}
\begin{align*}
\W (\RPow h 0) &= 0 \\
\W (\RPow h (1+n)) &= \Size{h} \times \W(\RPow_n) + \W(h) + \Size{h}^{1+n} \\
                   &= \Size{h} \times \W(\RPow_n) + O(\Size{\RPow_{1+n}}) \\
 &\therefore\\
\W (\RPow_n) &= O(\Size{\RPow_n}^2) \\
\end{align*}
\hrule
\begin{align*}
\end{align*}
}

\framet{Left-associated trees}{
\begin{textblock}{157}[1,0](350,5)
\setlength\mathindent{0.5ex}
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
\vspace{4ex}
\begin{align*}
\W (\LPow_0) &= 0 \\
\W (\LPow_{1+n}) &= \W(\ParOne) + \W(\LPow_n) + \Size{\LPow_n} + 1 \\
                &= \W(\LPow_n) + O(n) \\
 &\therefore\\
\W (\LPow_n) &= O(n^2) \\
\end{align*}
\hrule
\begin{align*}
\D (\LPow_0) &= 0 \\
\D (\LPow_{1+n}) &= \max (0,\D(\LPow_n)) + 1\\
 &\therefore\\
\D (\LPow_n) &= O(n) \\
\end{align*}
}


\end{document}

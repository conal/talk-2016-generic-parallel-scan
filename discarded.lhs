%% -*- latex -*-

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

%%%%%%%%%%%%%%%%%


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


\circuit{$8$}{0}{lsums-lv8}{7}{7}
\circuit{$16$}{0}{lsums-lv16}{15}{15}
\circuit{$8+8$}{0}{lsums-p-lv8}{22}{8}

\circuit{$(5+11)$ (unoptimized)}{1}{lsums-lv5xlv11-no-hash-no-opt}{25}{11}
\circuit{$(5+11)$ (optimized)}{1}{lsums-lv5xlv11}{25}{11}


\circuit{$8+8$}{0}{lsums-p-lv8}{22}{8}
\circuit{$2 \times 8$}{0}{lsums-p-lv8}{22}{8}

%%%%

\circuit{Products: $a^{m+n} = a^m \times a^n$}{0}{lsums-lv16}{15}{15}

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


%if False
\framet{Divide and conquer?}{
\vspace{-2ex}
\wfig{2.2in}{circuits/lsums-lv5}
\vspace{-5ex}
\wfig{4.5in}{circuits/lsums-lv11}

\emph{Then what?}
}

\circuit{Divide and conquer?}{0}{lsums-lv5-lv11-unknown-no-hash}{14+?}{7+?}
%endif
\circuit{Divide and conquer --- unequal}{0}{lsums-lv5xlv11}{25}{11}


\framet{Some simple scans}{
\vspace{-1ex}
\begin{center}
\ccap{|U1|}{1.2in}{lsums-u}
\ccap{|Par1|}{1in}{lsums-i}
\end{center}
\pause
\begin{center}
\ccap{|Par1 :*: U1|}{1.4in}{lsums-1-0-no-hash-no-opt}
\ccap{|Par1 :*: U1| (optimized)}{1in}{lsums-1-0}
\end{center}
\pause
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


\circuit{$8$}{0}{lsums-lv8}{7}{7}
\circuit{$16$}{0}{lsums-lv16}{15}{15}
\circuit{$5+(5+6)$}{-1}{lsums-lv5-5-6-r}{30}{7}

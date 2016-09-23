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

\definecolor{statColor}{rgb}{0.5,0,0}

% To remove

\newcommand{\hstats}[2]{
\begin{textblock}{100}[1,0](348,13)
{\small \textcolor{statColor}{work: #1, depth: #2}}
\end{textblock}
}

\newcommand{\stats}[2]{
{\small \textcolor{statColor}{work: #1, depth: #2}}}

\newcommand\circuit[3]{
\wfig{4.5in}{circuits/#1}
\vspace{-4ex}
\begin{center}
\stats{#2}{#3}
\end{center}
}


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

\framet{Prefix sum (left scan)}{\hstats{15}{15}

\circuit{lsumsp-lv16}{15}{15}
}

\framet{Divide and conquer?}{
\vspace{3ex}
\circuit{lsumsp-lv8-and-lv8}{14}{7}
}

\framet{Divide and conquer \emph{with correction}}{
\circuit{lsumsp-p-lv8}{22}{8}
}

\framet{$5+11$}{
\vspace{3ex}
\circuit{lsumsp-lv5xlv11}{25}{11}
}

\framet{$(5+5)+6$}{
\circuit{lsumsp-lv5-5-6-l}{24}{6}
}

\framet{$5+(5+6)$}{
\vspace{-1ex}
\circuit{lsumsp-lv5-5-6-r}{30}{7}
}

\framet{$8+8$}{
\circuit{lsumsp-p-lv8}{22}{8}
}

\framet{$2 \times 8$}{
\circuit{lsumsp-p-lv8}{22}{8}
}

\framet{$8 \times 2$}{
\circuit{lsumsp-lv8-p}{22}{8}
}

\framet{$4 \times 4$}{
\circuit{lsumsp-lv4olv4}{24}{6}
}

\framet{$((2 \times 2) \times 2) \times 2$}{
\vspace{-1ex}
\circuit{lsumsp-lb4}{24}{6}
}

\framet{$2 \times (2 \times (2 \times 2))$}{
\vspace{-1ex}
\circuit{lsumsp-rb4}{32}{4}
}

\framet{$(2 \times 2) \times (2 \times 2)$}{
\vspace{-1ex}
\circuit{lsumsp-bush2}{32}{4}
}

%% Try the textblock

\end{document}

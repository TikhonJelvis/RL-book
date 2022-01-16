---
title: Bellman Equation
---

$$
\node{V}{Bittersweet}{V^{\pi}(s)} = \mathbb{E}_{\pi, p}[\node{G}{NavyBlue}{G_t|S_t = s}] \text{ for all } s \in \mathcal{S}
$$
\begin{tikzpicture}[overlay,remember picture,>=stealth,nodes={align=left,inner ysep=1pt},<-]

\path (V.north) ++ (0,1.5em) node[anchor=south east,color=Bittersweet!90]  (V_label){\textbf{value of $s$ with policy $\pi$}};
\draw [color=Bittersweet!87](V.north) |- ([xshift=-0.3ex,color=Bittersweet]V_label.south west);

\path (G.south) ++ (0,-1em) node[anchor=north west,color=NavyBlue!90] (G_label){return from state $s$};
\draw [color=NavyBlue!87](G.south) |- ([xshift=-0.3ex,color=NavyBlue]G_label.south east);
\end{tikzpicture}

\vspace{10ex}

$$
V^{\node{p1}{Bittersweet}{\pi}}(\node{start}{NavyBlue}{s}) = \mathbb{E}_{\node{p2}{Bittersweet}{\pi}, p}[G_t|\node{all1}{PineGreen}{S_t = s}] \text{ for all } \node{all2}{PineGreen}{s \in \mathcal{S}}
$$

\begin{tikzpicture}[overlay,remember picture,>=stealth,nodes={align=left,inner ysep=1pt},<-]

\path (start.north) ++ (0, 4ex) node[anchor=south east,color=NavyBlue] (start_label){\textbf{start state}};
\draw[color=NavyBlue](start.north) |- ([xshift=-0.3ex,color=Bittersweet]start_label.south west);

\path (all1.north) ++ (0.4em,4ex) node[anchor=south west,color=PineGreen] (all_label){\textbf{each state}};
\draw[<->,color=PineGreen!70] (all1.north) -- ++(0,4ex)  -| node[] {} (all2.north);

\path (p1.south) ++ (0.4em,-8ex) node[anchor=south west,color=Bittersweet] (p_label){\textbf{policy $\pi$}};
\draw[<->,color=Bittersweet!70] (p1.south) -- ++(0,-5ex)  -| node[] {} (p2.south);
\end{tikzpicture}

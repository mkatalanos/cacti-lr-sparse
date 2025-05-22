# Model
Let $\boldsymbol{X_1}, \ldots, \boldsymbol{X_F} \in \mathbb{R}^{M \times N}$ represent a sequence of
$F$ images, each with dimensions $M\times N$. Our goal is to reconstruct them
from a snapshot measurement. 

## Measurement operator
Rather than observing $\boldsymbol{X_f}$ directly, for $f=1, \ldots, F$, we observe a matrix $\boldsymbol{Y}
\in \mathbb{R}^{M \times (N + F -1)}$ obtained by consecutively shifting the
frames horizontally by one pixel and adding them up. In other words, $\boldsymbol{Y}$ is generated
algorithmically as
- Create a matrix of zeros: $\boldsymbol{Y} = \boldsymbol{0_{M \times (N + F - 1)}}$
- For each $f = 1, \ldots, F$, 
    - Pad $\boldsymbol{X_f}$ with $f-1$ (resp. $F-f$) columns of zeros on the left (resp. right):

      $\boldsymbol{\widetilde{X}_f} = \begin{bmatrix} \boldsymbol{0_{M \times (f-1)}} & \boldsymbol{X_f} & \boldsymbol{0_{M \times (F-f)}}\end{bmatrix}$

    - Add the result to $\boldsymbol{Y}$: $\quad\boldsymbol{Y} \leftarrow \boldsymbol{Y} + \boldsymbol{\widetilde{X}_f}$

We will denote this procedure as $\mathcal{H}: (\mathbb{R}^{M\times N})^F \to \mathbb{R}^{M \times (N + F -
1)}$, i.e.,
$$
\begin{align*}
  \boldsymbol{Y} = \mathcal{H}(\boldsymbol{X_1},\, \ldots,\, \boldsymbol{X_F})\,.
\end{align*}
$$
This measurement process is linear. So there exists a matrix $\boldsymbol{H} \in
\mathbb{R}^{(M(N + F - 1))\times MNF}$ such that
$$
  \begin{align*}
    \boldsymbol{y}
    =
    \boldsymbol{H}
    \boldsymbol{x}\,,
  \end{align*}
$$
where $\boldsymbol{y} = \text{vec}(\boldsymbol{Y})$ and $\boldsymbol{x} =
\text{vec}(\begin{bmatrix}\text{vec}(\boldsymbol{X_1}) & \cdots &
\text{vec}(\boldsymbol{X_F})\end{bmatrix})$, and $\text{vec}(\cdot)$ denotes
the column-major vectorization of a matrix, i.e., it stacks its columns into a
column vector.

We will assume that there are efficient algorithms to compute $\boldsymbol{H
z}$ for any vector $\boldsymbol{z} \in \mathbb{R}^{MNF}$, and to compute
$\boldsymbol{H}^\top \boldsymbol{g}$ for any vector $\boldsymbol{g} \in
\mathbb{R}^{M(N + F - 1)}$. [^1] 

## Assumption: known background, sparse foreground

We make the following assumption: each frame $\boldsymbol{X_f} \in \mathbb{R}^{M\times N}$ in the sequence can be written as
$$
\begin{align*}
  \boldsymbol{X_f} = \boldsymbol{B} + \boldsymbol{S_f}\,, \qquad\qquad f = 1, \ldots, F\,,
\end{align*}
$$
where $\boldsymbol{B}\in \mathbb{R}^{M\times N}$ is known, and $\boldsymbol{S_f} \in \mathbb{R}^{M\times
N}$ is unknown but sparse.

## Estimation problem

Because of linearity of $\mathcal{H}$, we can subtract $\boldsymbol{Y_B} := \mathcal{H}(\boldsymbol{B},
\ldots, \boldsymbol{B})$ from $\boldsymbol{Y} = \mathcal{H}(\boldsymbol{X_1},\, \ldots,\, \boldsymbol{X_F})$, yielding
$$
\begin{align*}
  \boldsymbol{Y_S} := \boldsymbol{Y} - \boldsymbol{Y_B} =
  \mathcal{H}(\boldsymbol{X_1} - \boldsymbol{B}, \ldots, \boldsymbol{X_F} -
  \boldsymbol{B}) = \mathcal{H}(\boldsymbol{S_1}, \ldots, \boldsymbol{S_F})\,.
\end{align*}
$$
Thus, $\boldsymbol{S_1}, \ldots, \boldsymbol{S_F}$ can be recovered by
leveraging their sparsity and any additional assumptions, e.g., 
$$
\begin{align*}
  (\boldsymbol{\widehat{S}_1}, \ldots, \boldsymbol{\widehat{S}_F})
  \in
  \begin{array}{cl}
  \underset{\boldsymbol{S_1}, \ldots, \boldsymbol{S_F}}{\arg\min}
  &
  \|\boldsymbol{S_1}\|_1 + \cdots + \|\boldsymbol{S_f}\|_1 
  + 
  r(\boldsymbol{1}_{1\times F} \otimes \boldsymbol{B}  + 
  \begin{bmatrix}\boldsymbol{S_1} & \cdots & \boldsymbol{S_F} \end{bmatrix})
  \\
  \text{s.t.}
  &
  \boldsymbol{Y_S} = \mathcal{H}(\boldsymbol{S_1}, \ldots, \boldsymbol{S_F})\,,
  \end{array}
\end{align*}
$$
where $r\,:\, (\mathbb{R}^{M\times N})^F \to \mathbb{R}$ is a regularizer applied to the full sequence, e.g., 
low-rank of patches, as in [DeSCI](https://doi.org/10.1109/TPAMI.2018.2873587).

Once this problem is solved, one can estimate each $\boldsymbol{X_f}$ by adding $\boldsymbol{B}$ to
$\boldsymbol{\widehat{S}_f}$, i.e., $\boldsymbol{\widehat{X}_f} = \boldsymbol{B} + \boldsymbol{\widehat{S}_f}$.

[^1]: Note that these operations need not necessarily be computed in vector form.


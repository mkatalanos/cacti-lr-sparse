---
title: README
date:  2025-05-20
vim:nospell:
---

- **Deadline:** 2025-07-01
- **Title:** Snapshot Compressive Imaging with Foreground-Background Separation

# Main idea
**Message/hypothesis:** Assuming that a snapshot image can be decomposed into background +
  foreground, such that 
  1. the background is known in advance;
  2. the background is mostly static throughout the sequence;
  3. the foreground is sparse,
  
  enables reconstructing sequences with higher quality and faster.

## Documents

- [Problem formulation](problem-formulation) describes the one-pixel
  shift measurement operator, formalizes the assumption on
  background-foreground separation, and describes an estimation problem.
- [ADMM for background-foreground](admm-for-background-foreground) applies ADMM
  to solve the estimation problem.

# Todo

- [ ] Find or create sequence of images, real or synthetic, satisfying [above hypothesis](# Main idea). @Shubham
- [ ] Write code in Python and Matlab to implement matrix-vector multiplication @Shubham
      of [one-pixel shift operator](https://doi.org/10.3390/photonics8020034) (forward and adjoint operators)
- [ ] To establish baselines, run on above images the following algorithms
    - [ ] [DeSCI](https://doi.org/10.1109/TPAMI.2018.2873587), [code](https://github.com/liuyang12/DeSCI) @Marios
    - [ ] [E2E-CNN](https://doi.org/10.1063/1.5140721), [code](https://github.com/mq0829/DL-CACTI) @Marios
    - [ ] [PnP-ADMM](https://doi.org/10.1063/1.5140721),
          [code](https://github.com/mq0829/DL-CACTI); see also
          [PnP-FFDNet](https://doi.org/10.1109/CVPR42600.2020.00152) 
    - [ ] [DE-GAP-FFDNet](https://doi.org/10.1609/aaai.v37i3.25475),
          [code](https://github.com/IndigoPurple/DEQSCI) [optional]
    - [ ] [EfficientSCI++](https://doi.org/10.1007/s11263-024-02101-y),
          [code](https://github.com/mcao92/EfficientSCI-plus-plus) [optional]
- [X] Develop algorithm to solve [@Joao]
    $$
    \begin{align*}
      \begin{array}{ll}
      \underset{\boldsymbol{S_1}, \ldots, \boldsymbol{S_F}}{\text{minimize}}
      &
      \|\boldsymbol{S_1}\|_1 + \cdots + \|\boldsymbol{S_F}\|_1 
      + 
      r(\boldsymbol{B}, \overline{\boldsymbol{S}})
      \\
      \text{subject to}
      &
      \boldsymbol{Y_S} = \mathcal{H}(\boldsymbol{S_1}, \ldots, \boldsymbol{S_F})
      \end{array}
    \end{align*}
    $$
    See [file](admm/admm.pdf)
- [ ] Implement [admm](admm/admm.pdf) in Python/Matlab
- [ ] Test implementation for simple problem against [CVXPY](https://www.cvxpy.org/) or [CVX](https://cvxr.com/cvx/)
- [ ] Apply ADMM to synthetic images and compare against DeSCI and other algorithms
- [ ] Apply ADMM to real images and compare against DeSCI and other algorithms



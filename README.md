- **Deadline:** 2025-07-01
- **Title:** Snapshot Compressive Imaging with Foreground-Background Separation

# Main idea
**Message/hypothesis:** Assuming that a snapshot image can be decomposed into background +
  foreground, such that 
  1. ~~the background is known in advance;~~
  2. the background is mostly static throughout the sequence;
  3. the foreground is sparse,
  
  enables reconstructing sequences with higher quality and faster.

## Documents

- ~~[Problem formulation](problem-formulation) describes the one-pixel
  shift measurement operator, formalizes the assumption on
  background-foreground separation, and describes an estimation problem.~~
- ~~[ADMM for background-foreground](admm-for-background-foreground) applies ADMM
  to solve the estimation problem.~~
- This [document](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf) formulates the
  reconstruction problem, applies ADMM to solve it, and details each step.

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
- [ ] Implement [ADMM](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf) in Python;
      don't forget to unit test each step
    - [ ] Skeleton of the algorithm @Shubham + @Marios
    - [ ] Update of S and B @Shubham
    - [ ] Update of L, U, V @Marios
- [ ] Test implementation for simple problem against [CVXPY](https://www.cvxpy.org/) or [CVX](https://cvxr.com/cvx/)
- [ ] Apply ADMM to synthetic images and compare against DeSCI and other algorithms
- [ ] Apply ADMM to real images and compare against DeSCI and other algorithms


# Alternative ideas 

## Regularizers

After inspecting and profiling the
[DeSCI](https://doi.org/10.1109/TPAMI.2018.2873587)
[code](https://github.com/liuyang12/DeSCI), it seems that the operation that
takes longer is the extraction of patches of the data cube into the *low-rank*
matrix. Although there are probably several improvements that can be done, this
will likely make the final algorithm quite slow.

1. An alternative is to use a **video denoiser**, e.g.,
   [FastDVDNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html)
   ([code](https://github.com/m-tassano/fastdvdnet)), just as in
   [PnP-FastDVDNet](https://doi.org/10.1109/TPAMI.2021.3099035).

2. Another alternative is to use other patch priors. See two of the examples in
   the
   [DeepInv](https://deepinv.github.io/deepinv/auto_examples/index.html#examples)
   package.

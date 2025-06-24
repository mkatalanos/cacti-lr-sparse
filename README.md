- **Deadline:** 2025-07-01
- **Title:** Snapshot Compressive Imaging with Foreground-Background Separation

# Main idea
**Message/hypothesis:** Assuming that a snapshot image can be decomposed into background +
  foreground, such that 
  1. background sequence is low-rank;
  2. the foreground is sparse and has patch similarity
  
  enables reconstructing sequences with higher quality and faster than DeSCI
  and is competitive with PnP approaches.

## Documents

- This [document](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf) formulates the
  reconstruction problem, applies ADMM to solve it, and details each step.

# Todo

- [X] Implement algorithm in Python [@Marios]
- [ ] Integrate into code one-pixel (or more) shift operator. 
      Note that if the mask is the same for all the frames, the matrix $H$ in
      [document](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf) will be block
      diagonal and can be represented with the Kronecker product $(I \otimes
      M)$. If $M$ is invertible, then $(I \otimes M)^{-1} = (I \otimes
      M^{-1})$. See https://en.wikipedia.org/wiki/Kronecker_product .
      [@Shubham]
- [ ] To establish baselines, run on same sequences the some of the following algorithms
    - [ ] [DeSCI](https://doi.org/10.1109/TPAMI.2018.2873587), [code](https://github.com/liuyang12/DeSCI) [@Marios]
    - [ ] [E2E-CNN](https://doi.org/10.1063/1.5140721), [code](https://github.com/mq0829/DL-CACTI) [@Marios]
    - [ ] [PnP-ADMM](https://doi.org/10.1063/1.5140721),
          [code](https://github.com/mq0829/DL-CACTI); see also
          [PnP-FFDNet](https://doi.org/10.1109/CVPR42600.2020.00152) 
    - [ ] [DE-GAP-FFDNet](https://doi.org/10.1609/aaai.v37i3.25475),
          [code](https://github.com/IndigoPurple/DEQSCI) [optional]
    - [ ] [EfficientSCI++](https://doi.org/10.1007/s11263-024-02101-y),
          [code](https://github.com/mcao92/EfficientSCI-plus-plus) [optional]
    - [ ] A very recent paper: https://doi.org/10.1109/TIP.2025.3579208. The
          sensing scheme looks mildly related to the pixel-shift idea.
- [ ] Create overleaf with paper template [@Marios, @Shubham]
- [ ] Start writing some sections of the paper; see [below](## Structure of the paper) for the structure.

## Structure of the paper

- Introduction (~1 page)
  - Generic motivation for high-speed imaging (see examples in refs above);
    should be as short as possible 
  - Point out that unsupervised approaches like DeSCI achieve good
    reconstruction performance compared to approaches based on deep learning,
    despite relying on no training data, but can be computationally expensive.
    Also, in some scenarios, deep learning approaches are not applicable, e.g.,
    speckle noise (https://doi.org/10.1364/OE.469422).
  - Start a new paragraph with a bold title **Problem statement**, e.g., using
    in the preamble
    ```
    \newcommand{\mypar}[1]{\bigskip\noindent {\bf #1.}}
    ```
    First state the main assumptions. Then, state the problem exactly: "Our goal is to design a
    reconstruction algorithm for snapshot compressive imaging that leverages
    the above assumptions to improve reconstruction quality and efficiency with
    respect to baselines like DeSCI [ref] and ...". Then, we briefly summarize
    our approach and key results.
- Related work (~1 column).
    See papers for examples. I would organize the section as follows:
    - CASSI, CACTI, and unsupervised approaches.
    - Plug-and-play (PnP) methods.
    - End-to-end architectures.
- Proposed approach (~2-2.5 pages). Summarize [document](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf). 
    
- Experimental results (1 page)
- Conclusions
- References (5th page)


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

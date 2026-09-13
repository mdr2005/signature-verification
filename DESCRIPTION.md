# Signature Verification Engine

An offline handwritten-signature verification system that determines whether
a questioned signature was written by the same hand as a set of genuine
reference signatures — fully offline, no cloud APIs, no data leaving the
machine.

**Background.** Offline (static) signature verification is a hard problem: the
dynamic information of the writing process is lost, and hand-crafted feature
extractors struggle to separate genuine signatures from skilled forgeries,
leaving the best literature systems at roughly 7% verification error.[1]
Hafemann, Sabourin & Oliveira proposed learning the representations directly
from signature images with Convolutional Neural Networks (SigNet), formulating
feature learning in a Writer-Independent way, and reported 1.72% Equal Error
Rate on GPDS-160 versus 6.97% for the prior state of the art, with features
that generalized across GPDS, MCYT, CEDAR and Brazilian PUC-PR datasets.[1]

**What this project adds.** The paper's author released a PyTorch
re-implementation of SigNet with pretrained weights and writer-dependent
classifier code.[2] However, its stock preprocessing pipeline silently fails
on real-world images — phone photos of documents, screenshots, small scans —
collapsing the signature to an invisible smudge or thresholding shadows and
paper edges instead of ink, with no error raised. This project contributes:

- **Custom preprocessing for real-world inputs** — illumination correction
  (median-filter background estimation), ink-component analysis with stroke
  geometry, and GPDS-geometry normalization (COM-centering on a 952×1360
  canvas before the stock resize/crop) so the network sees its training
  geometry. Calibrated with ink-fraction diagnostics; validated on known
  same-writer/different-writer pairs.
- **A verification protocol** following the paper's writer-dependent
  formulation[1] as implemented in the sigver package:[2] automatic outlier-
  reference detection (mean pairwise cosine < 0.40), a writer-dependent SVM
  trained on genuines vs random forgeries (RBF, class-balanced), leave-one-out
  cross-validation printed on every run, and dual-crop testing for photographed
  documents.
- **Measured results** (own validation): same-writer vs different-writer
  cosine separation of 0.74–0.80 vs 0.22–0.26 on known pairs; 95% leave-one-out
  genuine acceptance and 100% random-forgery rejection on a 21-reference
  real-world set.

**Stack.** Python, PyTorch, scikit-image, scikit-learn. Project code is MIT;
the upstream SigNet package is BSD-3-Clause and its pretrained weights were
trained on GPDS, which is restricted to non-commercial use.[2]

## Sources

[1] https://arxiv.org/abs/1705.05787 — Hafemann et al. 2017 — Learning Features for Offline Handwritten Signature Verification using Deep CNNs
[2] https://github.com/luizgh/sigver — luizgh/sigver — PyTorch SigNet implementation

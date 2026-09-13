# Signature Verification Engine

Offline handwritten-signature verification for real-world images (scans,
phone photos of documents, screenshots) — built on the pretrained **SigNet**
CNN from [`luizgh/sigver`](https://github.com/luizgh/sigver) (PyTorch).

Given a set of genuine reference signatures from one writer and a questioned
signature, the engine answers: **same writer or forgery?** — with no cloud
APIs, no data leaving the machine.

## Why this project exists

The stock sigver preprocessing silently destroys non-GPDS images. It assumes
~500px-tall GPDS-style scans; on a 180px phone screenshot it shrinks the
signature to an invisible ~30px smudge (ink fraction 0.0001 vs the expected
0.003–0.02) and the model returns garbage — **with no error raised**. On full
document photos, OTSU thresholding grabs shadows, paper edges, and printed
text instead of the signature.

This project fixes that with a custom preprocessing pipeline that makes any
real-world image look GPDS-like before it reaches the network, plus a
verification protocol with built-in self-diagnostics.

## Results

Validated end-to-end on the sigver repo's known same-writer / different-writer
pairs, then on a real-world set of 22 reference signatures:

- **Known pairs (pipeline certification):** same-writer cosine similarity
  0.74–0.80, different-writer 0.22–0.26 — clean separation restored.
- **Real-world set:** 21/22 references formed a tight writer cluster (one
  outlier auto-detected and excluded); leave-one-out CV accepted 95% of
  held-out genuines and rejected 100% of random forgeries.
- **Case result:** a questioned signature from a photographed document was
  classified genuine, consistently across both a generous and a tight crop.

## How it works

```
user image ──► custom preprocessing ──► SigNet CNN ──► 2048-d feature ─┐
                                            (150×220, GPDS geometry)  │
references ──┤                                                        ▼
            │                                        writer-dependent SVM (RBF)
            └──► outlier detection (mean pairwise      genuines vs random
                 cosine < 0.40 → excluded)            forgeries, balanced
                                                        + leave-one-out CV
                                                        ▼
                                              verdict + cosine context
```

### 1. Custom preprocessing (`scripts/custom_preprocess.py`)
- **Clean scans/screenshots** → `prepare_clean_image(path)`: tight ink-bbox
  crop (OTSU + speck removal), rescale to ~500px tall (GPDS native size),
  background whitening, COM-center on a 952×1360 white canvas, then the stock
  `resize_image`/`crop_center` to the model's 150×220 input.
- **Document photos** → `prepare_photo(path, region)`: median-filter
  illumination correction, threshold at 0.85× OTSU, keep components ≥150px
  (drops noise and shadows, keeps pen strokes).
- Both paths reuse the stock pipeline's own geometry functions, so the model
  sees exactly the geometry it was trained on.

### 2. Verification protocol (`scripts/verify.py`)
Follows Hafemann et al.'s writer-dependent protocol:
1. Extract SigNet features (2048-d) for all references.
2. Detect outlier references (mean pairwise cosine < 0.40 → likely a
   different writer mixed into the "references"; excluded automatically).
3. Train a writer-dependent SVM — genuines + random forgeries, RBF kernel,
   class-balanced — and cross-validate with leave-one-out over genuines.
4. Classify the questioned signature. For photos, test both a generous and a
   tight crop; a robust verdict agrees across both.
5. Report verdict, SVM margin, and cosine-similarity context vs the writer's
   own internal variation.

## Repository layout

```
├── README.md                        ← this file
├── scripts/
│   ├── custom_preprocess.py         # real-world image → SigNet input
│   └── verify.py                    # end-to-end verification runner
├── references/
│   └── preprocessing-pitfalls.md    # full diagnosis + calibration notes
├── requirements.txt
└── LICENSE (MIT)
```

## Setup

```bash
git clone --depth 1 https://github.com/luizgh/sigver.git   # base repo (model + data)
cd sigver
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python scikit-image scikit-learn tqdm
# pretrained SigNet weights (63 MB):
curl -sL "https://drive.google.com/uc?export=download&id=1l8NFdxSvQSLb2QTv71E6bKcTgvShKPpx" -o models/signet.pth

# add this project's scripts (or clone it next to the sigver repo):
cp -r /path/to/this/repo/scripts .
```

Then run from the sigver repo root (so `sigver` package, `models/signet.pth`
and `data/*.png` resolve):

```bash
# clean scan / screenshot questioned image:
.venv/bin/python -W ignore scripts/verify.py \
    --refs 'user_sigs/refs/*.png' --questioned user_sigs/questioned.png

# questioned image is a document photo (crop rows 540:1280, cols 140:800):
.venv/bin/python -W ignore scripts/verify.py \
    --refs 'user_sigs/refs/*.png' --questioned user_sigs/photo.png \
    --photo-region 540 1280 140 800 --core-region 540 1000 140 720
```

## Interpreting output

- **Different writers**: cosine 0.22–0.35 vs the reference cluster.
- **Same writer** (clean pairs): 0.74+; real-world sets can sit lower/wider —
  the SVM trained on the writer's own variation makes the call.
- **Borderline** (below the refs' internal mean): reported as "consistent with
  genuine" with moderate confidence; more references firm it up.
- LOO CV is printed every run: **95%+ genuine acceptance + 100% forgery
  rejection = trustworthy classifier**; anything less, don't trust the verdict.

## Engineering notes

Deep-dive in [`references/preprocessing-pitfalls.md`](references/preprocessing-pitfalls.md):
diagnosis of the stock-pipeline failure modes with measured ink-fraction
calibration tables, component-analysis recipe for finding the signature
region in document photos, torch 2.x / scikit-image 0.26 API gotchas, and the
full verification-protocol details.

## Disclaimer

Automated screening tool, not a legally binding forensic examination.

## Credits

- SigNet CNN & sigver repo: Hafemann et al. / luizgh — see upstream LICENSE.
- Custom preprocessing, verification protocol, and validation: this project.

## License

MIT

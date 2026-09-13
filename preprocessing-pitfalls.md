# Sigver preprocessing pitfalls and calibration (session-validated)

## Project lineage
`luizgh/sigver_wiwd` (Theano+Lasagne, original) → `luizgh/sigver` (PyTorch
re-implementation by the same author, per its README). sigver adds WD
classifiers, meta-learning, and adversarial countermeasures. Pretrained
models are linked only from the README (Google Drive).

## Why stock preprocessing fails on real-world images

`preprocess_signature(img, canvas_size=(952,1360))` pipeline:
tight-crop by ink bbox → COM-center on 952x1360 canvas → invert →
resize to 170x242 → center-crop 150x220.

- Small screenshots (~150-220px): the signature occupies a tiny fraction of
  the 952x1360 canvas, so the resize to 170x242 shrinks strokes to ~1px.
  Observed ink fraction after processing: 0.0001-0.0005 (GPDS samples:
  0.003-0.02). **No error or warning is raised.**
- Full document photos: OTSU on the whole photo grabs dark bands, shadows,
  paper edges, printed text. Also `Warning: cropping image` fires when the
  ink bbox (including junk) exceeds the canvas — a telltale sign.

## Calibration numbers (GPDS-like, stock pipeline)

| sample | ink bbox (rows x cols of 150x220) | ink fraction |
|---|---|---|
| a1.png | 76 x 134 | 0.003 |
| a2.png | 91 x 177 | ~0.009 |
| some_signature.png | 41 x 70 | 0.009 |
| b1.png | 105 x 125 | ~0.02 |

Native tight crops of those samples: 300-600px tall, 560-1200px wide.
Target geometry for custom preprocessing: ink height ~70-100 rows,
width ~110-150 cols, ink fraction 0.003-0.03.

## Custom pipeline that worked (validated on known pairs)

`prepare_clean_image(path)`: grayscale → OTSU ink mask (remove components
< 30px) → tight bbox crop → rescale so height ≈ 500px (cap 900px wide) →
OTSU background whitening (keep stroke grayscale) → COM-center on 952x1360
white canvas → stock `resize_image((170,242))` + `crop_center((150,220))`.

`prepare_photo(path, region)`: crop region → illumination-correct
(median filter disk(31) background estimate, divide, rescale to 200 mean) →
threshold at 0.85 × OTSU → keep components ≥ 150px → rebuild grayscale
strokes (cap darkness 120) on white → tight crop → if crop ≥ 400x600 treat
as native (skip rescale) → same canvas/resize/crop path.

Validation with repo's known pairs (a1/a2 same writer, b1/b2 same, a-b
different): same-writer cosine 0.737-0.801, different-writer 0.215-0.264.
This separation is what certifies the pipeline; run it before user images.

## Photo region identification without vision

ASCII-render the photo (grayscale → downsample → ' .:-=+*#%@' ramp, dark =
dense char; height ≈ width/2 in rows for typical aspect) to eyeball where
the signature sits. Then component-analyze the illumination-corrected image:
pen strokes have consistent thickness ≈ area/diagonal-length ~6-8px along
their whole path; paper edges/shadows form large thin sheets or solid
blobs. A long descending flourish can extend hundreds of px below the core
signature — include it in the region (cutting it off changes features).
Test both a generous and a tight crop; a robust verdict agrees across both.

## Library gotchas (torch 2.14 / skimage 0.26, Sep 2026)

- `torch.load` on the pretrained signet.pth: needs `weights_only=False`
  (legacy pickle with non-tensor objects).
- `morphology.median` moved → `from skimage.filters import median`.
- `remove_small_objects(min_size=...)` deprecated → use `max_size=` (note:
  new semantics remove objects ≤ value).
- `img_as_ubyte` on float arrays outside [0,1] raises — clip/astype manually.
- `morphology.remove_small_objects` emits FutureWarnings — run scripts with
  `-W ignore` to keep output readable.
- venv created while cwd was inside a nested dir landed at
  `<cwd>/.venv` — check where `.venv` actually is before `uv pip install`.
- Google Drive direct download worked via
  `curl -sL "https://drive.google.com/uc?export=download&id=<ID>"` with no
  confirmation wall for a 63MB model and ~1MB images; one "image" was
  actually a ZIP (check with `file`, extract with python `zipfile`).

## Verification protocol (paper's method)

1. Features: SigNet → 2048-d, L2-normalized, cosine similarity.
2. Outlier refs: mean pairwise cosine to other refs < 0.40 → exclude
   (one of 22 refs flagged this way in validation; the rest clustered
   0.58-0.72, weak pairs all involved the outlier).
3. Check for a genuine 2-writer split: KMeans(2) silhouette near 0.12 =
   one writer; a real split would be higher.
4. WD-SVM: genuines (label 1) vs random forgeries (other writers' samples
   — repo `data/*.png` works — plus excluded outliers), RBF kernel, C=10,
   `class_weight='balanced'`. Leave-one-out CV on genuines for reliability
   (95% acceptance observed), verify forgeries rejected 100%.
5. Classify questioned features; also report cosine context vs the
   reference cluster.

Observed score bands (real-world set): different writers 0.22-0.35;
questioned-genuine borderline 0.49-0.65 with ref-ref internal mean 0.64,
min 0.45. SVM accepted (+0.29 margin) on both crops.

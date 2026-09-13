#!/usr/bin/env python3
"""End-to-end signature verification: features, outlier detection, WD-SVM.

Usage (from the sigver repo root):
    .venv/bin/python -W ignore <skill-scripts-dir>/verify.py \
        --refs 'user_sigs/refs/*.png' --questioned user_sigs/questioned.png

    # questioned image is a document photo (crop rows 540:1280, cols 140:800):
    ... verify.py --refs 'refs/*.png' --questioned photo.png \
        --photo-region 540 1280 140 800 --core-region 540 1000 140 720

Protocol (Hafemann et al., writer-dependent):
  1. Extract SigNet features for all reference signatures.
  2. Detect outlier references (mean pairwise cosine < 0.40 -> likely a
     different writer among the "references"; excluded from the genuine set).
  3. Train a writer-dependent SVM (genuines + random forgeries, RBF,
     class-balanced) and cross-validate with leave-one-out over genuines.
  4. Classify the questioned signature (both a generous and a tight crop
     when regions are given — results should agree).
  5. Report verdict + cosine-similarity context.

Random forgeries default to the repo's own samples (data/a1,a2,b1,b2.png);
add more with --forger-pool.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.svm import SVC

# Make both the skill scripts dir (custom_preprocess) and the sigver repo
# root (the `sigver` package) importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from custom_preprocess import prepare_clean_image, prepare_photo  # noqa: E402

from sigver.featurelearning.models import SigNet  # noqa: E402

OUTLIER_THRESHOLD = 0.40
MIN_INK_FRACTION = 0.002


def feat_from_processed(model, processed):
    x = torch.from_numpy(processed).view(1, 1, 150, 220).float().div(255)
    with torch.no_grad():
        return model(x).numpy()[0]


def make_svm():
    return SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')


def main():
    ap = argparse.ArgumentParser(
        description='Verify a questioned signature against reference signatures')
    ap.add_argument('--refs', nargs='+', required=True,
                    help='genuine reference images (globs allowed, >=2 recommended)')
    ap.add_argument('--questioned', required=True, help='signature image to verify')
    ap.add_argument('--model', default='models/signet.pth',
                    help='pretrained SigNet weights (default models/signet.pth)')
    ap.add_argument('--forger-pool', nargs='*', default=None,
                    help='extra known-other-writer images for the random-forgery '
                         'pool (default: repo samples data/a1,a2,b1,b2.png)')
    ap.add_argument('--photo-region', nargs=4, type=int, default=None,
                    metavar=('ROW0 ROW1 COL0 COL1'),
                    help='questioned image is a document photo: generous crop '
                         '(rows ROW0:ROW1, cols COL0:COL1) containing the signature')
    ap.add_argument('--core-region', nargs=4, type=int, default=None,
                    metavar=('ROW0 ROW1 COL0 COL1'),
                    help='tighter crop of the questioned signature (core strokes, '
                         'without flourish) — tested as a second opinion')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f'ERROR: model not found: {args.model} (see SKILL.md Setup)')
    if not os.path.exists(args.questioned):
        sys.exit(f'ERROR: questioned image not found: {args.questioned}')

    ref_paths = []
    for r in args.refs:
        ref_paths.extend(sorted(glob.glob(r)))
    ref_paths = list(dict.fromkeys(ref_paths))
    if len(ref_paths) < 2:
        sys.exit('ERROR: need >= 2 reference signature images')

    model = SigNet().eval()
    sd, _, _ = torch.load(args.model, weights_only=False)
    model.load_state_dict(sd)

    # ---- reference features -------------------------------------------------
    ref_feats, ref_names, skipped = [], [], []
    for p in ref_paths:
        try:
            processed = prepare_clean_image(p)
            if (processed > 64).mean() < MIN_INK_FRACTION:
                skipped.append((os.path.basename(p), 'no visible ink after preprocessing'))
                continue
            ref_feats.append(feat_from_processed(model, processed))
            ref_names.append(os.path.basename(p))
        except Exception as e:
            skipped.append((os.path.basename(p), f'{type(e).__name__}: {e}'))
    if skipped:
        print('skipped reference images:')
        for name, why in skipped:
            print(f'  {name}: {why}')
    if len(ref_feats) < 2:
        sys.exit('ERROR: fewer than 2 usable references after preprocessing')
    R = np.array(ref_feats)
    n = len(R)
    print(f'references: {len(ref_names)} usable / {len(ref_paths)} given')

    # ---- outlier detection --------------------------------------------------
    normed = R / np.linalg.norm(R, axis=1, keepdims=True)
    S = normed @ normed.T
    np.fill_diagonal(S, 1)
    mean_sim = (S.sum(axis=1) - 1) / (n - 1)
    print('\nmean similarity of each reference to the others:')
    for name, s in sorted(zip(ref_names, mean_sim), key=lambda t: t[1]):
        flag = '  <-- OUTLIER' if s < OUTLIER_THRESHOLD else ''
        print(f'  {name}: {s:.4f}{flag}')
    outliers = mean_sim < OUTLIER_THRESHOLD
    print(f'\nmain cluster: {(~outliers).sum()} refs, outliers: {outliers.sum()}')
    if outliers.all():
        sys.exit('ERROR: every reference is an outlier — the reference set is '
                 'inconsistent; check that all references are from the same writer.')
    genuine = R[~outliers]

    # ---- random-forgery pool ------------------------------------------------
    pool = args.forger_pool
    if pool is None:
        pool = [p for p in ['data/a1.png', 'data/a2.png', 'data/b1.png', 'data/b2.png']
                if os.path.exists(p)]
    rf_feats = []
    for p in pool:
        try:
            rf_feats.append(feat_from_processed(model, prepare_clean_image(p)))
        except Exception as e:
            print(f'  (skipping forger sample {p}: {type(e).__name__})')
    forgers = np.array(rf_feats) if rf_feats else np.empty((0, R.shape[1]))
    if outliers.any():
        forgers = np.vstack([forgers, R[outliers]])
    if len(forgers) < 4:
        print(f'WARNING: only {len(forgers)} random forgeries in the pool; '
              'the SVM boundary will be weak. Add --forger-pool images.')

    # ---- WD-SVM + leave-one-out CV -----------------------------------------
    X = np.vstack([genuine, forgers])
    y = np.hstack([np.ones(len(genuine)), np.zeros(len(forgers))])
    clf = make_svm()
    clf.fit(X, y)

    correct = 0
    for i in range(len(genuine)):
        c = make_svm()
        c.fit(np.delete(X, i, axis=0), np.delete(y, i, axis=0))
        if c.predict([genuine[i]])[0] == 1:
            correct += 1
    print(f'\nLOO cross-validation: {correct}/{len(genuine)} held-out genuine refs '
          f'recognized as genuine ({correct / len(genuine) * 100:.0f}%)')
    forg_correct = int((clf.predict(forgers) == 0).sum()) if len(forgers) else 0
    print(f'random forgeries rejected by full classifier: {forg_correct}/{len(forgers)}')

    # ---- classify the questioned signature ----------------------------------
    if args.photo_region is not None:
        crops = [('FULL (generous crop) ', args.photo_region)]
        if args.core_region is not None:
            crops.append(('CORE (tight crop)   ', args.core_region))
    elif args.core_region is not None:
        crops = [('CORE (tight crop)   ', args.core_region)]
    else:
        crops = [('clean image         ', None)]

    qf_last = None
    for label, region in crops:
        try:
            if region is not None:
                processed = prepare_photo(args.questioned, region=tuple(region))
            else:
                processed = prepare_clean_image(args.questioned)
            qf = feat_from_processed(model, processed)
        except Exception as e:
            print(f'\nquestioned {label}: preprocessing failed '
                  f'({type(e).__name__}: {e})')
            continue
        if qf_last is None:
            qf_last = qf
        pred = clf.predict([qf])[0]
        dist = clf.decision_function([qf])[0]
        verdict = 'GENUINE (same writer)' if pred == 1 else 'FORGERY (different writer)'
        print(f'\nquestioned {label}: SVM decision = {verdict}')
        print(f'  decision_function = {dist:+.3f}  (negative -> forgery side)')

    if qf_last is None:
        sys.exit('ERROR: questioned signature could not be preprocessed')

    sims = normed @ (qf_last / np.linalg.norm(qf_last))
    gidx = ~outliers
    print(f'\ncosine similarity vs main-cluster refs: '
          f'mean={sims[gidx].mean():.4f} max={sims[gidx].max():.4f}')
    print(f'cosine similarity vs all refs: mean={sims.mean():.4f} max={sims.max():.4f}')


if __name__ == '__main__':
    main()

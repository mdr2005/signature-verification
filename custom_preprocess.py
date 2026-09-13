"""Custom preprocessing for user-provided signature images.

The stock sigver pipeline assumes GPDS-style scans where the signature is
~500px tall on a large canvas. Small screenshots and document photos break
it silently (see references/preprocessing-pitfalls.md). This module makes
user images look GPDS-like, then reuses the stock resize+crop so the model
sees its training geometry.

Usage:
    from custom_preprocess import prepare_clean_image, prepare_photo
    img150x220 = prepare_clean_image('ref.png')            # scans/screenshots
    img150x220 = prepare_photo('photo.jpg', region=(r0, r1, c0, c1))  # photos

Output: 150x220 float array, white strokes (0-255) on black, ready for
    x = torch.from_numpy(p).view(1,1,150,220).float().div(255)

Validated on the sigver repo's known pairs: same-writer cosine 0.74-0.80,
different-writer 0.22-0.26 (stock-equivalent separation).
"""
import numpy as np
from skimage import img_as_ubyte, filters, transform, morphology
from skimage.io import imread
from skimage.color import rgb2gray


def to_gray(path):
    img = imread(path)
    if img.ndim == 3:
        img = img[..., :3] if img.shape[2] >= 3 else img[..., 0]
        g = img_as_ubyte(rgb2gray(img))
    else:
        g = img_as_ubyte(img)
    return g


def tight_crop_ink(gray, dark_is_ink=True):
    """Crop grayscale image to the bounding box of ink pixels (otsu mask)."""
    t = filters.threshold_otsu(gray)
    mask = (gray < t) if dark_is_ink else (gray > t)
    # remove tiny specks
    mask = morphology.remove_small_objects(mask, 30)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None, None
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return gray[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def scale_to_model_size(stroke_img, target_h=500, target_w=900, native=False):
    """Make a tight-cropped grayscale signature (dark ink on white) look like
    GPDS training data, using the stock pipeline's own geometry.

    Stock: native GPDS signature (~500px tall) COM-centered on 952x1360
    canvas -> invert -> resize to 170x242 -> center crop 150x220.

    If `native` is True the crop is assumed to already be GPDS-sized and is
    pasted without rescaling (avoids blur from an unnecessary resample).
    Returns 150x220 float, white strokes on black.
    """
    from sigver.preprocessing.normalize import resize_image, crop_center

    if native:
        scaled = stroke_img.astype(np.uint8)
        nh, nw = scaled.shape
    else:
        h, w = stroke_img.shape
        s = min(target_h / h, target_w / w)
        nh, nw = max(2, int(round(h * s))), max(2, int(round(w * s)))
        scaled = transform.resize(stroke_img, (nh, nw), preserve_range=True,
                                  anti_aliasing=True).astype(np.uint8)

    # Stock-style background removal: lighter than otsu -> white
    t = filters.threshold_otsu(scaled)
    scaled[scaled > t] = 255

    # COM-center on a 952x1360 white canvas (stock semantics: COM of ink)
    canvas_h, canvas_w = 952, 1360
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    ink = scaled < t
    if ink.any():
        rr, cc = np.where(ink)
        r_com, c_com = rr.mean(), cc.mean()
    else:
        r_com, c_com = nh / 2, nw / 2
    top = int(round(canvas_h / 2 - r_com))
    left = int(round(canvas_w / 2 - c_com))
    t0, l0 = max(0, top), max(0, left)
    t1 = min(canvas_h, top + nh)
    l1 = min(canvas_w, left + nw)
    if t1 <= t0 or l1 <= l0:
        raise ValueError('signature placement out of canvas bounds')
    canvas[t0:t1, l0:l1] = scaled[t0 - top:t1 - top, l0 - left:l1 - left]

    # stock pipeline: invert (white strokes on black), resize, center crop
    inverted = 255 - canvas
    resized = resize_image(inverted, (170, 242))
    return crop_center(resized, (150, 220)).astype(float)


def prepare_clean_image(path):
    """For clean scans/screenshots: dark ink on light background."""
    gray = to_gray(path)
    crop, _ = tight_crop_ink(gray)
    if crop is None:
        raise ValueError(f'no ink found in {path}')
    return scale_to_model_size(crop)


def prepare_photo(path, region=None, min_area=150):
    """For photos of documents: illumination-correct, threshold, select ink
    components inside the signature region, keep grayscale stroke values."""
    gray = to_gray(path)
    if region is not None:
        r0, r1, c0, c1 = region
        work = gray[r0:r1, c0:c1].copy()
    else:
        work = gray.copy()
        r0 = c0 = 0

    # Illumination correction: estimate background with big median filter,
    # divide it out.
    from skimage.filters import median
    from skimage.morphology import disk
    bg = median(work, disk(31))
    norm = np.clip(work.astype(float) / np.maximum(bg, 1) * 200, 0, 255)

    t = filters.threshold_otsu(norm.astype(np.uint8))
    mask = norm < t * 0.85  # strokes must be clearly darker than background

    # keep only reasonably-sized components (drops noise, keeps signature
    # strokes and any remaining text)
    from skimage.measure import label, regionprops
    lab = label(mask)
    keep = np.zeros_like(mask)
    for r in regionprops(lab):
        if r.area >= min_area:
            keep[lab == r.label] = True
    if not keep.any():
        raise ValueError('no ink found in photo region')

    # rebuild grayscale strokes on white background
    out = np.full(norm.shape, 255.0)
    out[keep] = np.minimum(norm[keep], 120)  # keep stroke darkness
    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    crop = out[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    # If the crop is already close to GPDS native size, skip rescaling
    ch, cw = crop.shape
    use_native = ch >= 400 and cw >= 600
    return scale_to_model_size(crop, native=use_native)

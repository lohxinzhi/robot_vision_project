"""Image-only character extraction using filtering, thresholding and components."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation import connected_components, prepare_character_binary


DEFAULT_CONFIG = Path(__file__).with_name("chip_preprocessing.json")


def load_preprocessing(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text())


def filtered_binary(gray, settings):
    """Apply optional grayscale median filtering before the Task 6 pipeline."""
    size = settings.get("gray_median_size", 1)
    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError("gray_median_size must be a positive odd integer")
    filtered = cv2.medianBlur(gray, size)
    _, _, binary = prepare_character_binary(
        filtered, settings["smooth_size"], settings["background_size"],
        settings["offset"], settings["median_size"],
    )
    return binary


def extract_glyphs(gray, settings, connectivity):
    """Locate characters, then optionally refine each bounded character mask.

    Mild filtering locates separate characters. Stronger grayscale filtering
    supplies smoother recognition masks within those established windows.
    Component selection uses area and geometry alone.
    """
    segmentation = settings["segmentation"]
    min_area = segmentation["min_area"]
    if not isinstance(min_area, int) or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    binary = filtered_binary(gray, segmentation)
    labels, components = connected_components(binary, connectivity, min_area=min_area)
    if not components:
        raise ValueError("No character components found")

    refinement = settings.get("glyph_refinement")
    margin = refinement["box_margin"] if refinement else 0
    if not isinstance(margin, int) or margin < 0:
        raise ValueError("box_margin must be a nonnegative integer")
    refined = filtered_binary(gray, refinement) if refinement else None
    glyphs = []
    windows = []
    display_mask = np.zeros_like(binary)
    for component in components:
        left = max(0, component["x"] - margin)
        top = max(0, component["y"] - margin)
        right = min(gray.shape[1], component["x"] + component["width"] + margin)
        bottom = min(gray.shape[0], component["y"] + component["height"] + margin)
        if refinement:
            # The established window keeps adjacent characters separate when
            # stronger filtering connects their masks in the full ROI.
            local_labels, local_components = connected_components(
                refined[top:bottom, left:right], connectivity, min_area=1,
            )
            if not local_components:
                raise ValueError("Refinement erased a located character")
            largest = max(local_components, key=lambda item: item["area"])
            mask = local_labels == largest["label"]
        else:
            mask = labels[top:bottom, left:right] == component["label"]
        foreground = mask.astype(np.uint8) * 255
        display_mask[top:bottom, left:right] = np.maximum(
            display_mask[top:bottom, left:right], foreground,
        )
        glyphs.append(Image.fromarray(255 - foreground))
        windows.append({"x": left, "y": top, "width": right - left,
                        "height": bottom - top, "foreground_area": int(mask.sum())})
    return glyphs, components, windows, binary, display_mask

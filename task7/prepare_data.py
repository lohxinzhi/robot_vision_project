"""Audit character images, create grouped splits and compare preprocessing."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def preprocess_character(image, mode="pad", size=32):
    """Return float32 pixels in [0, 1], with bright character foreground."""
    if mode not in ("resize", "pad", "stretch"):
        raise ValueError("mode must be resize, pad or stretch")
    if size < 1:
        raise ValueError("size must be positive")
    foreground = 255 - np.asarray(image.convert("L"), dtype=np.uint8)
    if mode in ("pad", "stretch"):
        rows, columns = np.nonzero(foreground > 51)
        if len(rows) == 0:
            return np.zeros((size, size), dtype=np.float32)
        foreground = foreground[rows.min():rows.max() + 1,
                                columns.min():columns.max() + 1]
        if mode == "pad":
            side = max(foreground.shape)
            margin = max(2, round(side * 0.12))
            canvas = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.uint8)
            top = (len(canvas) - foreground.shape[0]) // 2
            left = (len(canvas) - foreground.shape[1]) // 2
            canvas[top:top + foreground.shape[0], left:left + foreground.shape[1]] = foreground
            foreground = canvas
    resized = Image.fromarray(foreground).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def inventory_images(data_root):
    """Read class subfolders and group images by decoded grayscale pixels."""
    records = []
    groups = defaultdict(list)
    sizes = Counter()
    classes = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        files = sorted(path for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            continue
        classes.append(folder.name)
        for path in files:
            with Image.open(path) as source:
                gray = source.convert("L")
                sizes[f"{gray.width}x{gray.height}"] += 1
                content = (gray.size, gray.tobytes())
            record = {"path": path.relative_to(data_root).as_posix(), "label": folder.name}
            records.append(record)
            groups[content].append(record)
    if not records:
        raise ValueError("No character images found in class subfolders")
    for group in groups.values():
        if len({record["label"] for record in group}) != 1:
            raise ValueError("Identical pixels have conflicting labels: " +
                             ", ".join(record["path"] for record in group))
    return records, dict(groups), classes, dict(sizes)


def select_exact_groups(groups, target):
    """Use subset sum to select whole duplicate groups of an exact total size."""
    reachable = {0: []}
    for index, group in enumerate(groups):
        for total, selected in list(reachable.items()):
            new_total = total + len(group)
            if new_total <= target and new_total not in reachable:
                reachable[new_total] = selected + [index]
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError(f"Cannot allocate exactly {target} images as whole duplicate groups")
    selected = set(reachable[target])
    chosen = [group for index, group in enumerate(groups) if index in selected]
    remaining = [group for index, group in enumerate(groups) if index not in selected]
    return chosen, remaining


def assign_partitions(groups, classes, seed):
    """Reserve exactly 25% per class for testing, then 20% of development for validation."""
    random = np.random.default_rng(seed)
    for label in classes:
        class_groups = [group for group in groups.values() if group[0]["label"] == label]
        random.shuffle(class_groups)
        count = sum(len(group) for group in class_groups)
        if count % 4:
            raise ValueError(f"Class {label} has {count} images; an exact 75/25 split needs a multiple of four")
        test_count = count // 4
        validation_count = round((count - test_count) * 0.2)
        test, remaining = select_exact_groups(class_groups, test_count)
        validation, train = select_exact_groups(remaining, validation_count)
        for partition, selected in [("train", train), ("validation", validation), ("test", test)]:
            for group in selected:
                for record in group:
                    record["split"] = partition


def export_preview(records, classes, data_root, output):
    """Show two training examples per class under all preprocessing choices."""
    figure, axes = plt.subplots(len(classes), 8, figsize=(15, len(classes) * 1.9), squeeze=False)
    titles = ["Original", "Resize 32x32", "Crop + pad 32x32", "Crop + stretch 32x32"] * 2
    for row, label in enumerate(classes):
        examples = [record for record in records
                    if record["label"] == label and record["split"] == "train"][:2]
        if len(examples) < 2:
            raise ValueError(f"Need two training images for preview of class {label}")
        for example_index, record in enumerate(examples):
            with Image.open(data_root / record["path"]) as source:
                images = [np.asarray(source.convert("L")),
                          preprocess_character(source, "resize"),
                          preprocess_character(source, "pad"),
                          preprocess_character(source, "stretch")]
            for mode_index, image in enumerate(images):
                column = example_index * 4 + mode_index
                axis = axes[row, column]
                axis.imshow(image, cmap="gray", vmin=0, vmax=255 if mode_index == 0 else 1)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(titles[column], fontsize=10)
                if column == 0:
                    axis.set_ylabel(f"Class {label}", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "preprocessing_examples.png", dpi=160)
    plt.close(figure)


def prepare_data(data_root, output, seed=5411):
    data_root = data_root.resolve()
    records, groups, classes, sizes = inventory_images(data_root)
    assign_partitions(groups, classes, seed)
    split_counts = Counter(record["split"] for record in records)
    class_counts = {}
    for label in classes:
        counts = Counter(record["split"] for record in records if record["label"] == label)
        total = sum(counts.values())
        assert counts["test"] * 4 == total
        class_counts[label] = {**dict(counts), "development": counts["train"] + counts["validation"],
                               "total": total}
    cross_partition = sum(len({record["split"] for record in group}) > 1
                          for group in groups.values())
    assert cross_partition == 0
    assert sum(split_counts.values()) == len(records)
    assert len({record["path"] for record in records}) == len(records)
    summary = {
        "seed": seed, "classes": classes, "total_images": len(records),
        "image_sizes": sizes, "split_counts": dict(split_counts),
        "development_images": split_counts["train"] + split_counts["validation"],
        "per_class": class_counts, "unique_pixel_groups": len(groups),
        "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "duplicate_extra_files": len(records) - len(groups),
        "cross_partition_duplicate_groups": cross_partition,
        "root_images_excluded": sorted(path.name for path in data_root.iterdir()
                                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
        "preprocessing": {"size": [32, 32], "dtype": "float32", "range": [0, 1],
                          "foreground": "bright", "modes": ["resize", "pad", "stretch"],
                          "pad_foreground_threshold": 51, "pad_margin_fraction": 0.12},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_split.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_preview(records, classes, data_root, output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset_2026"),
                        help="Dataset root containing class subfolders")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--seed", type=int, default=5411)
    arguments = parser.parse_args()
    summary = prepare_data(arguments.data, arguments.output, arguments.seed)
    print(json.dumps(summary, indent=2))

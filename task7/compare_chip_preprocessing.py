"""Compare Task 6 binary characters with further image processing."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chip_processing import extract_glyphs, load_preprocessing
from recognize_chip import recognize


def compare(data, output, crop):
    settings = load_preprocessing()
    task6 = {
        "segmentation": {"gray_median_size": 1, "smooth_size": 3, "background_size": 31,
                         "offset": 4, "median_size": 5, "min_area": 500},
        "glyph_refinement": None,
    }
    cases = [
        ("task6_binary", "Task 6 binary characters", task6),
        ("refined", "Further mask refinement", settings),
    ]
    records = []
    rows = []
    for name, description, processing in cases:
        result = recognize(data, output, crop, processing, save=False)
        records.append({"id": name, "description": description, "result": result})
        for connectivity, item in result["connectivity"].items():
            row = {"case": name, "connectivity": connectivity, "count": item["character_count"]}
            for method, prediction in item["models"].items():
                row[method] = prediction["prediction"]
                row[f"{method}_accuracy"] = prediction["position_accuracy"]
            rows.append(row)
            print(f"{name}, {connectivity}-connected: CNN {row['cnn']}, SVM {row['svm']}", flush=True)

    evidence = {
        "applied_case": "refined",
        "comparison": "Task 6 binary characters versus grayscale median, local threshold and bounded component refinement; saved classifiers are shared",
        "experiments": records,
    }
    (output / "chip_preprocessing_experiments.json").write_text(json.dumps(evidence, indent=2) + "\n")
    with (output / "chip_preprocessing_experiments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Chip recognition using Task 6 masks and further image processing", "",
             "Case 1 directly classifies the binary character masks produced by Task 6. "
             "Case 2 applies grayscale median filtering, local thresholding and connected-component refinement within character windows. "
             "Both cases use the saved CNN/SVM classifiers and their recorded normalization.", "",
             "| Case | CNN, 4-connectivity | CNN, 8-connectivity | SVM, 4-connectivity | SVM, 8-connectivity |",
             "|---|---|---|---|---|"]
    latex = [r"\begin{center}\begin{tabular}{lll}\toprule",
             r"Input & CNN (4 / 8 connectivity) & SVM (4 / 8 connectivity) \\\midrule"]
    for record in records:
        markdown_cells = []
        latex_cells = []
        for method in ("cnn", "svm"):
            prediction_lines = []
            for connectivity in ("4", "8"):
                prediction = record["result"]["connectivity"][connectivity]["models"][method]
                score = prediction["position_accuracy"]
                accuracy = f"{100 * score:.0f}" if score is not None else "--"
                markdown_cells.append(f"`{prediction['prediction']}` ({accuracy}%)")
                prediction_lines.append(f"{connectivity} \\texttt{{{prediction['prediction']}}} ({accuracy}\\%)")
            latex_cells.append(r"\shortstack[l]{" + r" \\ ".join(prediction_lines) + "}")
        lines.append(f"| {record['description']} | " + " | ".join(markdown_cells) + " |")
        latex.append(record["description"] + " & " + " & ".join(latex_cells) + r" \\")
    latex.append(r"\bottomrule\end{tabular}\end{center}")
    lines += ["", "## Applied processing", "",
              "Task 6 uses a 3×3 mean, 31×31 local mean with offset 4, 5×5 binary median and minimum component area 500. "
              "Each retained component supplies a binary character mask directly to the classifier.", "",
              "Character location uses a 3×3 mean, 31×31 local mean with offset 2, 5×5 binary median and minimum component area 500. "
              "Each located window is expanded by two pixels. "
              "The recognition mask uses a 13×13 grayscale median, 3×3 mean, 31×31 local mean with offset 2 and 5×5 binary median. "
              "The largest connected region is retained inside each window. Parameters are in `../chip_preprocessing.json`.", "",
              "Rows labeled 4 and 8 in the LaTeX table denote the connectivity definitions. "
              "The reference `HD44780A00` scores each sequence after prediction. "
              "The chip image supplied development feedback, and the reported score covers its ten characters.", ""]
    (output / "chip_preprocessing_experiments.md").write_text("\n".join(lines))
    (output / "chip_preprocessing_experiments.tex").write_text("\n".join(latex) + "\n")

    image = cv2.imread(str(data), cv2.IMREAD_GRAYSCALE)
    x, y, width, height = crop
    gray = image[y:y + height, x:x + width]
    before, *_ = extract_glyphs(gray, task6, 8)
    after, *_ = extract_glyphs(gray, settings, 8)
    columns = max(len(before), len(after))
    figure, axes = plt.subplots(2, columns, figsize=(17, 5), squeeze=False)
    for row_index, glyphs in enumerate((before, after)):
        predictions = records[row_index]["result"]["connectivity"]["8"]["models"]
        for index, glyph in enumerate(glyphs):
            axis = axes[row_index, index]
            axis.imshow(glyph, cmap="gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            cnn = predictions["cnn"]["prediction"][index]
            svm = predictions["svm"]["prediction"][index]
            axis.set_title(f"{index + 1}\nCNN {cnn} / SVM {svm}", fontsize=9)
        for axis in axes[row_index, len(glyphs):]:
            axis.axis("off")
    axes[0, 0].set_ylabel("Task 6 binary")
    axes[1, 0].set_ylabel("Further refinement")
    figure.tight_layout()
    figure.savefig(output / "chip_preprocessing_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=here / "dataset_2026/Test_Image_2026.bmp")
    parser.add_argument("--output", type=Path, default=here / "results")
    parser.add_argument("--crop", type=int, nargs=4, default=(36, 202, 912, 136))
    arguments = parser.parse_args()
    compare(arguments.image, arguments.output, arguments.crop)

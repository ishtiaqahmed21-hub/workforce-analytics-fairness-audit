"""Render a clearly labeled dashboard preview from executed result artifacts."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / "reports/results.json").read_text())
    kpis = result["hr_kpis"]
    fig = plt.figure(figsize=(16, 10), facecolor="#eef4f5")
    fig.text(
        0.035,
        0.95,
        "WORKFORCE EVIDENCE STUDIO",
        color="#12344a",
        size=23,
        weight="bold",
    )
    fig.text(
        0.035,
        0.91,
        "DASHBOARD PROTOTYPE | Derived from fictional IBM HR records; Adult audit is a separate population",
        color="#526879",
        size=12,
    )
    values = [
        ("Sample records", f"{kpis['sample_records']:,}"),
        ("Attrition label share", f"{kpis['attrition_share']:.1%}"),
        ("Held-out HR F1", f"{result['hr']['test']['f1']:.3f}"),
        ("Held-out HR recall", f"{result['hr']['test']['recall']:.3f}"),
    ]
    for i, (label, value) in enumerate(values):
        ax = fig.add_axes([0.035 + i * 0.245, 0.77, 0.23, 0.10], facecolor="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.06, 0.69, label, color="#526879", size=12)
        ax.text(0.06, 0.15, value, color="#007f82", size=27, weight="bold")
    for rect, name in [
        ([0.035, 0.39, 0.56, 0.33], "attrition_by_role.png"),
        ([0.62, 0.39, 0.35, 0.33], "compensation.png"),
        ([0.035, 0.08, 0.56, 0.28], "fairness_selection.png"),
        ([0.62, 0.08, 0.35, 0.28], "adult_calibration.png"),
    ]:
        ax = fig.add_axes(rect)
        ax.imshow(plt.imread(ROOT / "figures" / name))
        ax.axis("off")
    fig.text(
        0.035,
        0.035,
        "IBM HR: ODbL 1.0 / DbCL 1.0. UCI Adult: CC BY 4.0. No annual turnover, causal effect or employment decision is represented.",
        size=10,
        color="#526879",
    )
    fig.savefig(ROOT / "dashboard/dashboard_prototype.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

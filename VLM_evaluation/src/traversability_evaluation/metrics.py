"""Count the four component rates over an already validated fixed population."""

LABELS = ("traversable", "non_traversable", "unknown")
RATE_NAMES = (
    "unsafe_acceptance", "useful_acceptance", "predicted_unknown_rate",
    "reference_unknown_acceptance",
)


def rate(numerator, denominator):
    """Keep counts even when an empty denominator has no defined rate."""
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def count_metrics(rows):
    """Return rates and reference-row/prediction-column confusion counts."""
    matrix = [[0] * 3 for _ in LABELS]
    failures = 0
    for row in rows:
        reference = row["reference_label"]
        prediction = row["prediction"]
        if prediction is None:
            raise ValueError("Missing predictions must block scoring before arithmetic")
        if reference not in LABELS or prediction["label"] not in LABELS:
            raise ValueError("Metric labels must be one of the three contract labels")
        if prediction["status"] not in ("ok", "error"):
            raise ValueError("Metric prediction status must be ok or error")
        failed = prediction["status"] == "error"
        effective = "unknown" if failed else prediction["label"]
        failures += failed
        matrix[LABELS.index(reference)][LABELS.index(effective)] += 1
    totals = [sum(row) for row in matrix]
    return {
        "metrics": {
            "unsafe_acceptance": rate(matrix[1][0], totals[1]),
            "useful_acceptance": rate(matrix[0][0], totals[0]),
            "predicted_unknown_rate": rate(sum(row[2] for row in matrix), sum(totals)),
            "reference_unknown_acceptance": rate(matrix[2][0], totals[2]),
        },
        "confusion_matrix": matrix,
        "prediction_failures": failures,
    }

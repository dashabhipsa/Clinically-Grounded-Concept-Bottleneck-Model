"""Tests for VinDr-CXR annotation parsing and concept vector creation."""
from src.data.annotations import (
    Box,
    concepts_from_boxes,
    group_boxes,
    image_labels_to_multihot,
    load_boxes,
    load_image_labels,
    normalize_class_name,
)

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES


def test_normalize_class_name():
    assert normalize_class_name("Pleural effusion") == "pleural_effusion"
    assert normalize_class_name("Nodule/Mass") == "nodule_mass"
    assert normalize_class_name("Lung opacity") == "lung_opacity"
    assert normalize_class_name("ILD") == "ild"
    assert normalize_class_name(" Aortic enlargement ") == "aortic_enlargement"


def test_load_boxes_columns(synthetic_vindr):
    df = load_boxes(synthetic_vindr / "annotations" / "train.csv")
    expected = {"image_id", "class_id", "class_name", "x_max", "x_min", "y_max", "y_min", "rad_id"}
    assert expected.issubset(set(df.columns))


def test_group_boxes(synthetic_vindr):
    df = load_boxes(synthetic_vindr / "annotations" / "train.csv")
    by_image = group_boxes(df)
    assert by_image["0001"]  # non-empty
    for box in by_image["0001"]:
        assert isinstance(box, Box)
        assert box.image_id == "0001"
    # 0001 has cardiomegaly + pleural effusion boxes
    classes = {b.canonical_class for b in by_image["0001"]}
    assert classes == {"cardiomegaly", "pleural_effusion"}


def test_box_normalized_coordinates(synthetic_vindr):
    df = load_boxes(synthetic_vindr / "annotations" / "train.csv")
    by_image = group_boxes(df)
    for box in by_image["0001"]:
        assert 0.0 <= box.x_min <= box.x_max <= 1.0
        assert 0.0 <= box.y_min <= box.y_max <= 1.0
        assert box.width >= 0.0
        assert box.height >= 0.0


def test_concepts_from_boxes_multihot(synthetic_vindr):
    df = load_boxes(synthetic_vindr / "annotations" / "train.csv")
    image_ids = ["0001", "0002", "0003", "0004", "0005", "0006"]
    matrix = concepts_from_boxes(df, CONCEPT_NAMES, image_ids=image_ids)
    assert list(matrix.columns) == CONCEPT_NAMES
    assert list(matrix.index) == image_ids

    row_0001 = matrix.loc["0001"].to_dict()
    assert row_0001["cardiomegaly"] == 1
    assert row_0001["pleural_effusion"] == 1
    # edema is not a VinDr-CXR class -> always 0
    assert row_0001["edema"] == 0
    assert row_0001["lung_opacity"] == 0

    row_0002 = matrix.loc["0002"].to_dict()
    assert row_0002["lung_opacity"] == 1
    assert row_0002["cardiomegaly"] == 0


def test_concepts_missing_findings_all_zero(synthetic_vindr):
    df = load_boxes(synthetic_vindr / "annotations" / "train.csv")
    matrix = concepts_from_boxes(df, CONCEPT_NAMES, image_ids=["nonexistent"])
    assert (matrix.loc["nonexistent"] == 0).all()


def test_image_labels_to_multihot(synthetic_vindr):
    labels = load_image_labels(synthetic_vindr / "annotations" / "image_labels_train.csv")
    matrix = image_labels_to_multihot(labels, DIAGNOSIS_NAMES)
    assert matrix.loc["0001", "cardiomegaly"] == 1
    assert matrix.loc["0001", "pleural_effusion"] == 1
    assert matrix.loc["0002", "lung_opacity"] == 1
    # 'consolidation' absent for 0001
    assert matrix.loc["0001", "consolidation"] == 0
    # requested classes not present in the labels are zero-filled
    assert matrix.loc["0001", "pulmonary_fibrosis"] == 0

"""Tests for porkqc.labels."""

import pytest

from porkqc.labels import Box, load_class_names, read_label_file, split_stats, yolo_to_box


def test_yolo_to_box_converts_to_pixels():
    b = yolo_to_box("2 0.5 0.5 0.2 0.4", 100, 50)
    assert b == Box(2, 40.0, 15.0, 60.0, 35.0)
    assert b.width == pytest.approx(20) and b.height == pytest.approx(20)


def test_yolo_to_box_full_image():
    b = yolo_to_box("0 0.5 0.5 1 1", 720, 540)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 720, 540)
    assert b.area == pytest.approx(720 * 540)


@pytest.mark.parametrize(
    "line",
    ["0 0.5 0.5 0.2", "0 0.1 0.2 0.3 0.4 0.5 0.6", "0 1.5 0.5 0.2 0.2", "0 0.5 0.5 -0.1 0.2"],
)
def test_yolo_to_box_rejects_invalid_lines(line):
    with pytest.raises(ValueError):
        yolo_to_box(line, 100, 100)


def test_yolo_to_box_rejects_bad_image_size():
    with pytest.raises(ValueError):
        yolo_to_box("0 0.5 0.5 0.2 0.2", 0, 100)


def test_missing_label_file_means_no_objects(tmp_path):
    assert read_label_file(tmp_path / "missing.txt", 720, 540) == []


def test_read_label_file_skips_blank_lines(tmp_path):
    f = tmp_path / "img.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n\n1 0.2 0.2 0.1 0.1\n")
    boxes = read_label_file(f, 720, 540)
    assert [b.cls for b in boxes] == [0, 1]


def test_load_class_names_list_and_dict(tmp_path):
    f = tmp_path / "data.yaml"
    f.write_text("names: ['a', 'b']\n")
    assert load_class_names(f) == ["a", "b"]
    f.write_text("names:\n  1: b\n  0: a\n")
    assert load_class_names(f) == ["a", "b"]


def test_split_stats(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.1 0.1\n")
    (tmp_path / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "c.txt").write_text("")
    s = split_stats(tmp_path)
    assert s["n_images"] == 3 and s["n_empty"] == 1 and s["n_boxes"] == 3
    assert s["class_counts"][0] == 2 and s["images_per_class"][0] == 2
    assert s["rel_areas"][0] == pytest.approx(0.25)

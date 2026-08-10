"""Class mapping is complete, stable and bijective."""

import pytest

from ai.preprocessing import labels


def test_seven_classes_in_stable_order():
    assert labels.NUM_CLASSES == 7
    assert labels.CLASS_CODES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def test_mapping_is_bijective():
    mapping = labels.class_to_index()
    assert sorted(mapping.values()) == list(range(labels.NUM_CLASSES))
    for index, code in enumerate(labels.index_to_class()):
        assert mapping[code] == index


def test_every_code_has_a_clinical_name():
    assert set(labels.CLASS_NAMES) == set(labels.CLASS_CODES)


def test_class_to_index_returns_a_copy():
    labels.class_to_index()["akiec"] = 99
    assert labels.encode("akiec") == 0


def test_unknown_code_rejected():
    with pytest.raises(ValueError, match="Unknown class code"):
        labels.encode("melanoma")

    with pytest.raises(ValueError, match="unknown class codes"):
        labels.validate_codes(["nv", "not_a_class"])

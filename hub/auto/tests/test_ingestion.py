from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.exceptions import InvalidPathException, SamePathException
import pytest
import hub


def test_ingestion_simple(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")

    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            dest_creds=None,
            compression="jpeg",
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(
            src=path, dest=path, dest_creds=None, compression="jpeg", overwrite=False
        )

    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        dest_creds=None,
        compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.info.class_names == ("class0", "class1", "class2")


def test_image_classification_sets(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        dest_creds=None,
        compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3,)
    assert ds["test/labels"].info.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3,)
    assert ds["train/labels"].info.class_names == ("class0", "class1", "class2")


def test_ingestion_exception(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            dest_creds=None,
            compression="jpeg",
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(
            src=path, dest=path, dest_creds=None, compression="jpeg", overwrite=False
        )

from pathlib import Path

import pytest

from benchmark_data.adapters.xstest import XSTestAdapter
from benchmark_data.schema import BenchmarkSample


def test_load_real_dataset():
    adapter = XSTestAdapter()
    samples = adapter.load()

    assert len(samples) > 0
    sample = samples[0]

    assert isinstance(sample, BenchmarkSample)
    assert sample.benchmark == "XSTest"
    assert sample.prompt
    assert sample.sample_id.startswith("xstest_")
    assert sample.category

    assert "original_id" in sample.metadata
    assert "label" in sample.metadata
    assert "focus" in sample.metadata
    assert "note" in sample.metadata
    assert sample.metadata["dataset_name"] == "paul-rottger/xstest"


def test_missing_file_raises(tmp_path):
    adapter = XSTestAdapter(data_path=tmp_path / "tidak_ada.csv")
    with pytest.raises(FileNotFoundError):
        adapter.load()


def test_missing_required_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "id,prompt,type,label,focus\n"
        "1,contoh prompt,homonyms,safe,kill\n",
        encoding="utf-8",
    )
    adapter = XSTestAdapter(data_path=bad_csv)
    with pytest.raises(ValueError):
        adapter.load()


def test_empty_dataset_raises(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text(
        "id,prompt,type,label,focus,note\n",
        encoding="utf-8",
    )
    adapter = XSTestAdapter(data_path=empty_csv)
    with pytest.raises(ValueError):
        adapter.load()


def test_repr():
    """
    Memastikan representasi object XSTestAdapter benar.
    """

    data_path = Path("data/raw/xstest/xstest_prompts.csv")

    adapter = XSTestAdapter(data_path=data_path)

    expected = (
        "XSTestAdapter("
        f"data_path='{data_path}')"
    )

    assert repr(adapter) == expected


def main():
    print("Menjalankan XSTestAdapter tests...")
    test_load_real_dataset()
    print("✓ Load real dataset")
    print("\nSemua XSTestAdapter test berhasil.")


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import Iterable, Optional

from benchmark_data.schema import BenchmarkSample
from evaluation.experiment_executor import ExperimentExecutor
from evaluation.raw_result_writer import RawExperimentResultWriter


class BatchExperimentRunner:
    """
    Menjalankan eksperimen untuk banyak sample.

    Fitur:
        - Persistent model lifecycle melalui batch_session().
        - Resume berdasarkan sample_id dalam raw JSONL.
        - Skip sample yang sudah selesai.
        - Menyimpan hasil segera setelah sample selesai.
        - Melanjutkan batch ketika satu sample gagal.
        - Pemilihan sample opsional berdasarkan sample_ids.

    Kompatibilitas:
        runner.run(samples)
        runner.run()

    run() tanpa argumen menggunakan dataset_samples milik
    ExperimentExecutor, jika atribut tersebut tersedia.
    """

    def __init__(
        self,
        executor: ExperimentExecutor,
        output_path: Path,
        resume: bool = True,
        sample_ids: Optional[Iterable[str]] = None,
    ):
        self.executor = executor
        self.output_path = Path(output_path)
        self.resume = resume

        # None berarti semua sample akan diproses.
        # Daftar kosong berarti tidak ada sample yang dipilih.
        self.sample_ids = (
            None
            if sample_ids is None
            else list(dict.fromkeys(str(sid) for sid in sample_ids))
        )

        self.writer = RawExperimentResultWriter(
            self.output_path
        )

    def _load_completed_sample_ids(self) -> set[str]:
        """
        Membaca sample_id dari raw result JSONL untuk resume.
        """

        if not self.output_path.exists():
            return set()

        completed_ids: set[str] = set()

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Raw result JSONL tidak valid "
                        f"pada baris {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        "Record raw result harus berupa object JSON "
                        f"pada baris {line_number}."
                    )

                sample_id = record.get("sample_id")

                if sample_id is None or str(sample_id).strip() == "":
                    raise ValueError(
                        "Record raw result tidak memiliki "
                        f"sample_id pada baris {line_number}."
                    )

                completed_ids.add(str(sample_id))

        return completed_ids

    def _resolve_samples(
        self,
        samples: Optional[Iterable[BenchmarkSample]],
    ) -> list[BenchmarkSample]:
        """
        Menentukan sample yang akan dijalankan.

        Prioritas:
        1. samples yang diberikan langsung ke run(samples).
        2. executor.dataset_samples jika run() tanpa argumen.

        Jika sample_ids ditentukan, hanya sample dengan ID tersebut
        yang dipilih. ID yang tidak ditemukan akan memunculkan error.
        """

        if samples is None:
            samples = getattr(
                self.executor,
                "dataset_samples",
                None,
            )

        if samples is None:
            raise ValueError(
                "Daftar sample tidak tersedia. Panggil "
                "runner.run(samples), atau pastikan executor "
                "memiliki atribut dataset_samples."
            )

        available_samples = list(samples)

        # Tanpa filter berarti semua sample dipakai.
        if self.sample_ids is None:
            return available_samples

        samples_by_id: dict[str, BenchmarkSample] = {}

        for sample in available_samples:
            sample_id = str(sample.sample_id)

            if sample_id in samples_by_id:
                raise ValueError(
                    "Ditemukan sample_id duplikat dalam dataset: "
                    f"{sample_id}"
                )

            samples_by_id[sample_id] = sample

        missing_ids = [
            sample_id
            for sample_id in self.sample_ids
            if sample_id not in samples_by_id
        ]

        if missing_ids:
            raise ValueError(
                "sample_ids yang diminta tidak ditemukan dalam "
                f"dataset: {missing_ids}"
            )

        # Mengikuti urutan sample_ids yang diminta.
        return [
            samples_by_id[sample_id]
            for sample_id in self.sample_ids
        ]

    def _validate_result(
        self,
        *,
        expected_sample_id: str,
        result,
    ) -> None:
        """
        Memastikan hasil eksperimen valid sebelum disimpan.
        """

        if result is None:
            raise ValueError(
                "ExperimentExecutor mengembalikan result=None."
            )

        if str(result.sample_id) != str(expected_sample_id):
            raise ValueError(
                "sample_id hasil eksperimen tidak cocok: "
                f"expected={expected_sample_id}, "
                f"actual={result.sample_id}"
            )

        if not result.complete:
            raise ValueError(
                "ExperimentResult belum lengkap untuk "
                f"sample_id={expected_sample_id}."
            )

    def run(
        self,
        samples: Optional[Iterable[BenchmarkSample]] = None,
    ) -> dict[str, int]:
        """
        Menjalankan batch experiment.

        Pemakaian:
            runner.run(dataset_samples)
            runner.run()

        Sample yang sudah ada dalam output dilewati jika
        resume=True.
        """

        selected_samples = self._resolve_samples(samples)

        completed_ids = (
            self._load_completed_sample_ids()
            if self.resume
            else set()
        )

        total = len(selected_samples)
        skipped = 0
        completed = 0
        failed = 0

        print("=" * 60)
        print("BATCH EXPERIMENT")
        print("=" * 60)
        print(f"Total sample : {total}")
        print(f"Resume       : {self.resume}")
        print(f"Output       : {self.output_path}")
        print("=" * 60)

        # Jika tidak ada sample terpilih, tidak perlu memuat model.
        if total == 0:
            summary = {
                "total": 0,
                "skipped": 0,
                "completed": 0,
                "failed": 0,
            }
            print("Tidak ada sample untuk diproses.")
            return summary

        try:
            with self.executor.batch_session() as runner:
                for index, sample in enumerate(
                    selected_samples,
                    start=1,
                ):
                    sample_id = str(sample.sample_id)

                    print(
                        f"\n[{index}/{total}] "
                        f"sample_id={sample_id}"
                    )

                    # Resume: lewati sample yang sudah tercatat.
                    if (
                        self.resume
                        and sample_id in completed_ids
                    ):
                        print(
                            "  -> SKIP "
                            "(hasil sudah tersedia)"
                        )
                        skipped += 1
                        continue

                    try:
                        result = self.executor.run_sample_loaded(
                            runner,
                            sample_id,
                        )

                        self._validate_result(
                            expected_sample_id=sample_id,
                            result=result,
                        )

                        # Pastikan direktori output tersedia.
                        self.output_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        # Simpan segera setelah sample selesai.
                        self.writer.write(result)

                        completed_ids.add(sample_id)
                        completed += 1

                        print("  -> COMPLETED")

                    except Exception as exc:
                        failed += 1

                        print("  -> FAILED")
                        print(
                            f"     {type(exc).__name__}: {exc}"
                        )

                        # Kegagalan satu sample tidak menghentikan batch.
                        continue

        except Exception:
            print("\nBATCH SESSION GAGAL.")
            raise

        summary = {
            "total": total,
            "skipped": skipped,
            "completed": completed,
            "failed": failed,
        }

        print("\n" + "=" * 60)
        print("BATCH SUMMARY")
        print("=" * 60)
        print(f"Total     : {summary['total']}")
        print(f"Skipped   : {summary['skipped']}")
        print(f"Completed : {summary['completed']}")
        print(f"Failed    : {summary['failed']}")
        print("=" * 60)

        return summary

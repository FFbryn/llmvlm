import json
from pathlib import Path
from typing import Iterable

from benchmark_data.schema import BenchmarkSample
from evaluation.experiment_executor import ExperimentExecutor
from evaluation.raw_result_writer import RawExperimentResultWriter


class BatchExperimentRunner:
    """
    Menjalankan eksperimen untuk banyak sample.

    Fitur utama:

        - persistent model lifecycle
        - resume dari raw JSONL
        - skip sample yang sudah selesai
        - menyimpan hasil segera setelah sample selesai
        - tidak menghentikan seluruh batch ketika satu sample gagal

    Lifecycle model dikendalikan oleh:

        ExperimentExecutor.batch_session()

    sehingga model di-load sekali dan digunakan untuk
    seluruh sample dalam satu batch.
    """

    def __init__(
        self,
        executor: ExperimentExecutor,
        output_path: Path,
        resume: bool = True,
    ):
        self.executor = executor
        self.output_path = Path(output_path)
        self.resume = resume

        self.writer = RawExperimentResultWriter(
            self.output_path
        )

    def _load_completed_sample_ids(self) -> set[str]:
        """
        Membaca sample_id yang sudah tersimpan
        pada raw result JSONL.

        Digunakan untuk mekanisme resume.
        """

        if not self.output_path.exists():
            return set()

        completed_ids: set[str] = set()

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
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

                sample_id = record.get("sample_id")

                if not sample_id:
                    raise ValueError(
                        "Record raw result tidak memiliki "
                        f"sample_id pada baris {line_number}."
                    )

                completed_ids.add(str(sample_id))

        return completed_ids

    def _validate_result(
        self,
        *,
        expected_sample_id: str,
        result,
    ) -> None:
        """
        Memastikan hasil eksperimen valid sebelum
        ditulis ke persistent storage.
        """

        if result is None:
            raise ValueError(
                "ExperimentExecutor mengembalikan "
                "result=None."
            )

        if result.sample_id != expected_sample_id:
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
        samples: Iterable[BenchmarkSample],
    ) -> dict[str, int]:
        """
        Menjalankan batch experiment.

        Model lifecycle:

            batch_session()
                ↓
            load LLM
            load VLM
                ↓
            sample 1
            sample 2
            sample 3
            ...
                ↓
            unload VLM
            unload LLM

        Sample yang sudah terdapat pada output JSONL
        akan dilewati apabila resume=True.
        """

        samples = list(samples)

        completed_ids = (
            self._load_completed_sample_ids()
            if self.resume
            else set()
        )

        total = len(samples)
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

        try:
            with self.executor.batch_session() as runner:

                for index, sample in enumerate(
                    samples,
                    start=1,
                ):
                    sample_id = sample.sample_id

                    print(
                        f"\n[{index}/{total}] "
                        f"sample_id={sample_id}"
                    )

                    # -------------------------------------
                    # Resume
                    # -------------------------------------

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

                    # -------------------------------------
                    # Run
                    # -------------------------------------

                    try:
                        result = (
                            self.executor.run_sample_loaded(
                                runner,
                                sample_id,
                            )
                        )

                        self._validate_result(
                            expected_sample_id=sample_id,
                            result=result,
                        )

                        # ---------------------------------
                        # Persistent write
                        # ---------------------------------

                        self.writer.write(result)

                        completed_ids.add(sample_id)
                        completed += 1

                        print(
                            "  -> COMPLETED"
                        )

                    except Exception as exc:
                        failed += 1

                        print(
                            "  -> FAILED"
                        )
                        print(
                            f"     {type(exc).__name__}: "
                            f"{exc}"
                        )

                        # Jangan hentikan batch.
                        continue

        except Exception:
            """
            Exception pada batch_session sendiri tidak
            dianggap sebagai sample failure.

            Misalnya model gagal di-load, maka seluruh
            batch tidak dapat dilanjutkan.
            """

            print(
                "\nBATCH SESSION GAGAL."
            )

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
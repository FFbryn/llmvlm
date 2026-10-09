import json
from pathlib import Path
from typing import Iterable, Optional, Set

from benchmark_data.schema import BenchmarkSample
from evaluation.experiment_executor import ExperimentExecutor
from evaluation.raw_result_writer import RawExperimentResultWriter


class BatchExperimentRunner:
    """
    Menjalankan eksperimen untuk banyak sample.

    Fitur:
    - Menjalankan sample secara berurutan.
    - Melanjutkan eksperimen menggunakan resume.
    - Hanya melewati sample yang memiliki tiga kondisi lengkap.
    - Memvalidasi hasil sebelum menyimpannya.
    - Melaporkan jumlah sample yang selesai, dilewati, dan gagal.

    Catatan:
    - Resume memeriksa kelengkapan struktur hasil mentah.
    - Classification tidak diwajibkan karena dilakukan pada tahap terpisah.
    - File output yang sudah ada tidak dihapus secara otomatis.
    """

    REQUIRED_CONDITIONS = (
        "llm_text",
        "vlm_text",
        "vlm_image",
    )

    def __init__(
        self,
        executor: ExperimentExecutor,
        output_path: str | Path,
        resume: bool = True,
        sample_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.executor = executor
        self.output_path = Path(output_path)
        self.resume = resume

        self.sample_ids = (
            set(sample_ids) if sample_ids is not None else None
        )

        self.writer = RawExperimentResultWriter(self.output_path)

    def _load_completed_sample_ids(self) -> Set[str]:
        """
        Membaca JSONL dan mengidentifikasi sample yang benar-benar
        memiliki ketiga kondisi eksperimen dengan struktur valid.

        File yang berisi JSON rusak, record tidak lengkap, atau ID
        duplikat akan menghasilkan error agar masalah data tidak
        tersembunyi.
        """
        completed_ids: Set[str] = set()

        if not self.output_path.exists():
            return completed_ids

        with self.output_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "JSONL tidak valid pada "
                        f"{self.output_path}, baris {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        "Record JSONL harus berupa object/dict pada "
                        f"baris {line_number}."
                    )

                sample_id = record.get("sample_id")

                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(
                        "sample_id tidak valid pada "
                        f"{self.output_path}, baris {line_number}."
                    )

                # Pastikan semua kondisi tersedia dan berbentuk object.
                for condition in self.REQUIRED_CONDITIONS:
                    condition_result = record.get(condition)

                    if not isinstance(condition_result, dict):
                        raise ValueError(
                            f"Record sample {sample_id!r} pada baris "
                            f"{line_number} tidak lengkap: "
                            f"{condition!r} harus berupa object/dict. "
                            "Jangan gunakan record ini untuk resume."
                        )

                    nested_sample_id = condition_result.get("sample_id")

                    if nested_sample_id != sample_id:
                        raise ValueError(
                            f"sample_id pada kondisi {condition!r} "
                            f"tidak cocok dengan sample_id utama "
                            f"untuk sample {sample_id!r}, "
                            f"baris {line_number}."
                        )

                if sample_id in completed_ids:
                    raise ValueError(
                        f"sample_id duplikat {sample_id!r} ditemukan "
                        f"pada {self.output_path}, baris {line_number}. "
                        "Gunakan salinan file yang sudah diaudit atau "
                        "output baru setelah memeriksa hasil sebelumnya."
                    )

                completed_ids.add(sample_id)

        return completed_ids

    def _resolve_samples(
        self,
        samples: Optional[Iterable[BenchmarkSample]] = None,
    ) -> list[BenchmarkSample]:
        """
        Menentukan sample yang akan dijalankan.
        """
        source_samples = (
            list(samples)
            if samples is not None
            else list(self.executor.dataset_samples)
        )

        seen_ids: Set[str] = set()

        for sample in source_samples:
            sample_id = sample.sample_id

            if sample_id in seen_ids:
                raise ValueError(
                    f"sample_id duplikat dalam dataset: {sample_id!r}"
                )

            seen_ids.add(sample_id)

        if self.sample_ids is None:
            return source_samples

        available_ids = {
            sample.sample_id for sample in source_samples
        }

        missing_ids = self.sample_ids - available_ids

        if missing_ids:
            raise ValueError(
                "Sample ID yang diminta tidak ditemukan dalam dataset: "
                f"{sorted(missing_ids)}"
            )

        return [
            sample
            for sample in source_samples
            if sample.sample_id in self.sample_ids
        ]

    @staticmethod
    def _validate_result(
        sample: BenchmarkSample,
        result,
    ) -> None:
        """
        Memastikan hasil eksperimen memiliki tiga kondisi lengkap.
        """
        if result is None:
            raise ValueError(
                f"Eksperimen sample {sample.sample_id!r} "
                "menghasilkan None."
            )

        if result.sample_id != sample.sample_id:
            raise ValueError(
                "sample_id hasil tidak cocok: "
                f"diharapkan {sample.sample_id!r}, "
                f"didapat {result.sample_id!r}."
            )

        if not result.complete:
            missing_conditions = [
                condition
                for condition in BatchExperimentRunner.REQUIRED_CONDITIONS
                if getattr(result, condition, None) is None
            ]

            raise ValueError(
                f"Hasil sample {sample.sample_id!r} belum lengkap. "
                f"Kondisi yang belum tersedia: {missing_conditions}"
            )

        # Periksa kecocokan ID pada setiap kondisi.
        for condition in BatchExperimentRunner.REQUIRED_CONDITIONS:
            condition_result = getattr(result, condition)

            if condition_result.sample_id != sample.sample_id:
                raise ValueError(
                    f"sample_id pada kondisi {condition!r} "
                    f"tidak cocok untuk sample {sample.sample_id!r}."
                )

    def run(
        self,
        samples: Optional[Iterable[BenchmarkSample]] = None,
    ) -> dict:
        """
        Menjalankan batch eksperimen dan mengembalikan ringkasan.
        """
        resolved_samples = self._resolve_samples(samples)

        completed_ids = (
            self._load_completed_sample_ids()
            if self.resume
            else set()
        )

        total = len(resolved_samples)
        skipped = 0
        completed = 0
        failed = 0
        failures = []

        # Jangan otomatis menghapus atau menimpa output lama.
        # Jika resume=False, gunakan output_path baru yang kosong.
        if not self.resume and self.output_path.exists():
            if self.output_path.stat().st_size > 0:
                raise FileExistsError(
                    f"Output sudah ada dan tidak kosong: "
                    f"{self.output_path}. Untuk menjalankan batch baru "
                    "dengan resume=False, gunakan path output baru "
                    "agar hasil sebelumnya tidak tercampur."
                )

        with self.executor.batch_session() as runner:
            for sample in resolved_samples:
                sample_id = sample.sample_id

                if self.resume and sample_id in completed_ids:
                    skipped += 1
                    print(f"[SKIP] {sample_id}: hasil lengkap sudah ada.")
                    continue

                try:
                    result = self.executor.run_sample_loaded(
                        runner,
                        sample_id,
                    )

                    self._validate_result(sample, result)

                    self.writer.write(result)
                    completed_ids.add(sample_id)
                    completed += 1

                    print(
                        f"[OK] {sample_id}: "
                        "llm_text, vlm_text, vlm_image selesai."
                    )

                except Exception as exc:
                    failed += 1
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "error": str(exc),
                        }
                    )

                    print(f"[ERROR] {sample_id}: {exc}")

        summary = {
            "total": total,
            "skipped": skipped,
            "completed": completed,
            "failed": failed,
            "failures": failures,
            "output_path": str(self.output_path),
        }

        print("\n" + "=" * 60)
        print("BATCH EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total samples : {total}")
        print(f"Skipped       : {skipped}")
        print(f"Completed     : {completed}")
        print(f"Failed        : {failed}")
        print(f"Output        : {self.output_path}")
        print("=" * 60)

        return summary
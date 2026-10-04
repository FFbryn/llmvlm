from dataclasses import dataclass
from pathlib import Path

from evaluation.batch_experiment_runner import (
    BatchExperimentRunner,
)


@dataclass
class DummyResult:
    sample_id: str
    complete: bool = True


class DummyRunner:
    def __init__(self):
        self.loaded = False
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        if not self.loaded:
            self.loaded = True
            self.load_count += 1

    def unload(self):
        if self.loaded:
            self.loaded = False
            self.unload_count += 1


class DummyThreeConditionRunner:
    def __init__(self):
        self.llm_pipeline = type(
            "Pipeline",
            (),
            {
                "runner": DummyRunner(),
            },
        )()

        self.vlm_pipeline = type(
            "Pipeline",
            (),
            {
                "runner": DummyRunner(),
            },
        )()


class DummyExecutor:
    def __init__(self):
        self.runner = DummyThreeConditionRunner()
        self.generated_sample_ids = []

    class _BatchSession:
        def __init__(self, executor):
            self.executor = executor

        def __enter__(self):
            llm_runner = (
                self.executor.runner
                .llm_pipeline.runner
            )

            vlm_runner = (
                self.executor.runner
                .vlm_pipeline.runner
            )

            llm_runner.load()
            vlm_runner.load()

            return self.executor.runner

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            vlm_runner = (
                self.executor.runner
                .vlm_pipeline.runner
            )

            llm_runner = (
                self.executor.runner
                .llm_pipeline.runner
            )

            vlm_runner.unload()
            llm_runner.unload()

            return False

    def batch_session(self):
        return self._BatchSession(self)

    def run_sample_loaded(
        self,
        runner,
        sample_id,
    ):
        if not (
            runner.llm_pipeline.runner.loaded
            and runner.vlm_pipeline.runner.loaded
        ):
            raise RuntimeError(
                "Dummy runner belum loaded."
            )

        self.generated_sample_ids.append(
            sample_id
        )

        return DummyResult(
            sample_id=sample_id,
            complete=True,
        )


class DummyWriter:
    def __init__(self):
        self.results = []

    def write(self, result):
        self.results.append(result)


class DummySample:
    def __init__(self, sample_id):
        self.sample_id = sample_id


def test_batch_runner_uses_persistent_lifecycle(
    tmp_path: Path,
):
    executor = DummyExecutor()

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=tmp_path / "results.jsonl",
        resume=False,
    )

    writer = DummyWriter()
    batch_runner.writer = writer

    samples = [
        DummySample("sample_1"),
        DummySample("sample_2"),
        DummySample("sample_3"),
    ]

    summary = batch_runner.run(samples)

    llm_runner = (
        executor.runner.llm_pipeline.runner
    )

    vlm_runner = (
        executor.runner.vlm_pipeline.runner
    )

    assert summary["total"] == 3
    assert summary["completed"] == 3
    assert summary["failed"] == 0
    assert summary["skipped"] == 0

    assert executor.generated_sample_ids == [
        "sample_1",
        "sample_2",
        "sample_3",
    ]

    assert llm_runner.load_count == 1
    assert llm_runner.unload_count == 1

    assert vlm_runner.load_count == 1
    assert vlm_runner.unload_count == 1

    assert llm_runner.loaded is False
    assert vlm_runner.loaded is False

    assert len(writer.results) == 3


def test_batch_runner_resumes_existing_results(
    tmp_path: Path,
):
    executor = DummyExecutor()

    output_path = (
        tmp_path / "results.jsonl"
    )

    output_path.write_text(
        '{"sample_id": "sample_1"}\n',
        encoding="utf-8",
    )

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=True,
    )

    writer = DummyWriter()
    batch_runner.writer = writer

    samples = [
        DummySample("sample_1"),
        DummySample("sample_2"),
        DummySample("sample_3"),
    ]

    summary = batch_runner.run(samples)

    assert summary["total"] == 3
    assert summary["skipped"] == 1
    assert summary["completed"] == 2
    assert summary["failed"] == 0

    assert executor.generated_sample_ids == [
        "sample_2",
        "sample_3",
    ]


def test_batch_runner_can_disable_resume(
    tmp_path: Path,
):
    executor = DummyExecutor()

    output_path = (
        tmp_path / "results.jsonl"
    )

    output_path.write_text(
        '{"sample_id": "sample_1"}\n',
        encoding="utf-8",
    )

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=False,
    )

    writer = DummyWriter()
    batch_runner.writer = writer

    samples = [
        DummySample("sample_1"),
        DummySample("sample_2"),
    ]

    summary = batch_runner.run(samples)

    assert summary["total"] == 2
    assert summary["skipped"] == 0
    assert summary["completed"] == 2
    assert summary["failed"] == 0

    assert executor.generated_sample_ids == [
        "sample_1",
        "sample_2",
    ]


def test_batch_runner_continues_after_sample_failure(
    tmp_path: Path,
):
    class FailingExecutor(DummyExecutor):
        def run_sample_loaded(
            self,
            runner,
            sample_id,
        ):
            if sample_id == "sample_2":
                raise RuntimeError(
                    "simulated failure"
                )

            return super().run_sample_loaded(
                runner,
                sample_id,
            )

    executor = FailingExecutor()

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=tmp_path / "results.jsonl",
        resume=False,
    )

    writer = DummyWriter()
    batch_runner.writer = writer

    samples = [
        DummySample("sample_1"),
        DummySample("sample_2"),
        DummySample("sample_3"),
    ]

    summary = batch_runner.run(samples)

    assert summary["total"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["skipped"] == 0

    assert executor.generated_sample_ids == [
        "sample_1",
        "sample_3",
    ]

    llm_runner = (
        executor.runner.llm_pipeline.runner
    )

    vlm_runner = (
        executor.runner.vlm_pipeline.runner
    )

    assert llm_runner.load_count == 1
    assert llm_runner.unload_count == 1

    assert vlm_runner.load_count == 1
    assert vlm_runner.unload_count == 1

    assert llm_runner.loaded is False
    assert vlm_runner.loaded is False
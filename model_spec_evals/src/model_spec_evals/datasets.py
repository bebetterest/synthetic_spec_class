from __future__ import annotations

from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, json_dataset


def model_spec_eval_dataset(
    dataset_dir: str,
    section_filter: list[str] | None = None,
    focus_filter: list[str] | None = None,
) -> Dataset:
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' does not exist. "
            "Pass -T dataset_dir=/path/to/dataset when running the task."
        )

    dataset_files = [
        path
        for path in sorted(dataset_path.iterdir())
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    ]
    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset files found in '{dataset_path}'. "
            "Expected files ending with .json or .jsonl."
        )

    datasets = [
        json_dataset(json_file=str(path), name=path.name) for path in dataset_files
    ]
    memory_dataset = MemoryDataset(samples=[sample for ds in datasets for sample in ds])

    memory_dataset = memory_dataset.filter(
        lambda sample: sample.metadata is not None
        and not sample.metadata.get("skip", False)
    )

    if section_filter:
        memory_dataset = memory_dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata["section_id"] in section_filter
        )

    if focus_filter:
        memory_dataset = memory_dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata["focus_id"] in focus_filter
        )

    return memory_dataset

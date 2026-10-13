from __future__ import annotations

from inspect_ai import Task, task

from model_spec_evals.datasets import model_spec_eval_dataset
from model_spec_evals.scorers import model_graded_spec_section_compliance


@task
def model_spec_section_task(
    dataset_dir: str | None = None,
    grader_model: str = "openai/gpt-5",
    num_grader_samples: int = 1,
    section_filter: list[str] | None = None,
    focus_filter: list[str] | None = None,
) -> Task:
    if dataset_dir is None:
        raise ValueError(
            "dataset_dir must be provided. Run with -T dataset_dir=/path/to/dataset."
        )
    return Task(
        dataset=model_spec_eval_dataset(
            dataset_dir=dataset_dir,
            section_filter=section_filter,
            focus_filter=focus_filter,
        ),
        scorer=model_graded_spec_section_compliance(
            model=grader_model, num_samples=num_grader_samples
        ),
        name="model_spec_evals",
    )

"""Run Tinker Cookbook supervised fine-tuning on conversation JSONL."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from training.checkpoints import load_run_info, write_run_info
from training.data import SFTPlan, SFTSettings, plan_sft, recorded_data, write_combined_jsonl
from training.infer import require_api_key, require_cookbook


def describe(plan: SFTPlan, renderer: str | None) -> str:
    settings = plan.settings
    renderer_text = renderer or "(recommended renderer, resolved when training starts)"
    checkpoint = settings.load_checkpoint or "(none)"
    if len(plan.sources) == 1:
        data_line = f"data: {plan.data}"
    else:
        shown = ", ".join(str(source) for source in plan.sources)
        data_line = f"data: {shown}\ncombined: {plan.data}"
    lines = [
        data_line,
        f"rows: {plan.rows} train={plan.train_rows} test={plan.test_rows}",
        f"log_dir: {settings.log_dir}",
        f"model: {settings.model}",
        f"renderer: {renderer_text}",
        f"learning_rate: {settings.learning_rate} schedule={settings.lr_schedule}",
        f"epochs: {settings.num_epochs} batch_size={settings.batch_size} max_length={settings.max_length}",
        f"lora_rank: {settings.lora_rank} train_on_what={settings.train_on_what}",
        f"save_every: {settings.save_every} eval_every: {settings.eval_every} max_steps: {settings.max_steps}",
        f"load_checkpoint: {checkpoint}",
        f"overwrite: {settings.overwrite}",
    ]
    remainder = plan.train_rows % settings.batch_size
    if remainder == 1:
        lines.append("note: the last training row does not fill a batch and is not trained on")
    elif remainder:
        lines.append(
            f"note: the last {remainder} training rows do not fill a batch and are not trained on"
        )
    return "\n".join(lines)


def resume_conflict(
    log_dir: Path,
    *,
    model: str,
    renderer: str,
    data: str | list[str],
    overwrite: bool,
) -> str | None:
    """Return an error when resume would continue a run under new settings.

    Resume keeps the optimizer state in the log directory, so the model,
    renderer, and dataset must be the ones that started it.
    """

    if overwrite:
        return None
    existing = load_run_info(log_dir)
    if not existing:
        return None
    same_model = existing.get("model") == model
    recorded_renderer = existing.get("renderer")
    same_renderer = recorded_renderer in (None, renderer)
    same_data = existing.get("data") == data
    if same_model and same_renderer and same_data:
        return None
    return (
        "This log dir belongs to "
        f"model={existing.get('model')!r} renderer={recorded_renderer!r} data={existing.get('data')!r}. "
        "Repeat those settings, or pass --overwrite to start over."
    )


def run_sft(settings: SFTSettings, dataset_builder: Any = None) -> SFTPlan:
    """Validate the dataset, then call ``tinker_cookbook.supervised.train.main``.

    Running the same command again resumes from the last ``state_path`` in
    ``log_dir/checkpoints.jsonl``. ``overwrite`` deletes that directory first.
    A ``load_checkpoint`` path is used only when the log directory has no
    resume checkpoint.

    ``dataset_builder`` replaces ``FromConversationFileBuilder``. The JSONL is
    still validated, and several inputs are still combined into
    ``log_dir/combined.jsonl`` before training starts. The builder is any
    object Tinker accepts as ``train.Config.dataset_builder``.
    """

    plan = plan_sft(settings)
    require_cookbook()
    require_api_key()

    import asyncio

    from tinker_cookbook import cli_utils, model_info
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer = settings.renderer or model_info.get_recommended_renderer_name(settings.model)
    model_info.warn_if_renderer_not_recommended(settings.model, renderer)
    behavior = "delete" if settings.overwrite else "resume"
    cli_utils.check_log_dir(str(settings.log_dir), behavior_if_exists=behavior)
    data_record = recorded_data(plan.sources)
    conflict = resume_conflict(
        settings.log_dir,
        model=settings.model,
        renderer=renderer,
        data=data_record,
        overwrite=settings.overwrite,
    )
    if conflict:
        raise ValueError(conflict)
    if len(plan.sources) > 1:
        write_combined_jsonl(plan.sources, plan.data)
    write_run_info(
        settings.log_dir,
        {
            "model": settings.model,
            "renderer": renderer,
            "data": data_record,
            "train_on_what": settings.train_on_what,
            "learning_rate": settings.learning_rate,
            "lr_schedule": settings.lr_schedule,
            "num_epochs": settings.num_epochs,
            "batch_size": settings.batch_size,
            "max_length": settings.max_length,
            "lora_rank": settings.lora_rank,
        },
    )
    print(describe(plan, renderer), file=sys.stderr)

    if dataset_builder is None:
        dataset_builder = FromConversationFileBuilder(
            file_path=str(plan.data),
            test_size=settings.test_size,
            shuffle_seed=settings.shuffle_seed,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=settings.model,
                renderer_name=renderer,
                max_length=settings.max_length,
                batch_size=settings.batch_size,
                train_on_what=TrainOnWhat(settings.train_on_what),
            ),
        )
    config = train.Config(
        log_path=str(settings.log_dir),
        model_name=settings.model,
        recipe_name="synthetic_spec_class_sft",
        load_checkpoint_path=settings.load_checkpoint,
        renderer_name=renderer,
        dataset_builder=dataset_builder,
        learning_rate=settings.learning_rate,
        lr_schedule=settings.lr_schedule,
        num_epochs=settings.num_epochs,
        lora_rank=settings.lora_rank,
        base_url=settings.base_url,
        save_every=settings.save_every,
        eval_every=settings.eval_every,
        wandb_project=settings.wandb_project,
        wandb_name=settings.wandb_name,
        max_steps=settings.max_steps,
    )
    asyncio.run(train.main(config))
    return plan

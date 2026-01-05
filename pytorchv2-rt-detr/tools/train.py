"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import time
from pathlib import Path
from typing import Iterable, Iterator

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from src import data as _data_registry  # noqa: F401 ensures transforms are registered


_CHECKPOINT_FILENAMES: tuple[str, ...] = ("last.pth", "checkpoint_last.pth", "best.pth")
_BACKBONE_CHECKPOINT_PREFERENCE: tuple[str, ...] = (
    "model_best.pth",
    "checkpoint_best.pth",
    "checkpoint_last.pth",
    "last.pth",
)

_DEFAULT_WEIGHTS_ROOT = "output/detr_compressed34"


def _prepare_run_directory(cfg: YAMLConfig, config_path: str, weights_root: str | None) -> None:
    if not weights_root:
        return
    if cfg.resume:
        return

    base_root = Path(weights_root).expanduser()
    try:
        base_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[output-dir] Warning: failed to create weights root {base_root}: {exc}")
        return

    config_stem = Path(config_path).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_root / f"{config_stem}_{timestamp}"

    cfg.output_dir = str(run_dir)
    cfg.yaml_cfg['output_dir'] = cfg.output_dir

    if not cfg.summary_dir:
        summary_path = run_dir / "summary"
        cfg.summary_dir = str(summary_path)
        cfg.yaml_cfg['summary_dir'] = cfg.summary_dir

    print(f"[output-dir] Writing run artifacts to {run_dir}")


def _iter_checkpoint_candidates(directory: Path) -> Iterable[Path]:
    for name in _CHECKPOINT_FILENAMES:
        candidate = directory / name
        try:
            if candidate.exists():
                yield candidate
        except OSError:
            continue


def _find_latest_checkpoint_for_config(config_path: str, output_dir: str | None,
                                       checkpoint_root: str | None) -> Path | None:
    config_stem = Path(config_path).stem
    config_token = config_stem.lower()
    search_targets: list[tuple[Path, bool]] = []

    if output_dir:
        search_targets.append((Path(output_dir).expanduser(), False))
        if checkpoint_root is None:
            search_targets.append((Path(output_dir).expanduser().parent, True))

    if checkpoint_root is not None:
        search_targets.append((Path(checkpoint_root).expanduser(), True))

    best_path: Path | None = None
    best_mtime: float | None = None
    visited: set[Path] = set()

    for base_dir, require_match in search_targets:
        try:
            resolved = base_dir.resolve()
        except OSError:
            resolved = base_dir
        if resolved in visited:
            continue
        visited.add(resolved)

        if not resolved.exists():
            continue

        if require_match:
            try:
                entries = list(resolved.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir():
                    continue
                if config_token not in entry.name.lower():
                    continue
                for candidate in _iter_checkpoint_candidates(entry):
                    try:
                        mtime = candidate.stat().st_mtime
                    except OSError:
                        continue
                    if best_mtime is None or mtime > best_mtime:
                        best_path = candidate
                        best_mtime = mtime
            continue

        for candidate in _iter_checkpoint_candidates(resolved):
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if best_mtime is None or mtime > best_mtime:
                best_path = candidate
                best_mtime = mtime

    return best_path


def _format_backbone_prefix(variant: str, coeff_luma: int, coeff_cb: int, coeff_cr: int) -> str:
    tag = f"coeffY{coeff_luma}"
    if coeff_cb == coeff_cr:
        tag += f"_CbCr{coeff_cb}"
    else:
        tag += f"_Cb{coeff_cb}_Cr{coeff_cr}"
    return f"{variant}_{tag}"


def _pick_best_checkpoint_in_dir(run_dir: Path) -> Path | None:
    for name in _BACKBONE_CHECKPOINT_PREFERENCE:
        candidate = run_dir / name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    best_path: Path | None = None
    best_mtime: float | None = None
    for candidate in run_dir.glob("*.pth"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if best_mtime is None or mtime > best_mtime:
            best_path = candidate
            best_mtime = mtime
    return best_path


def _iter_matching_backbone_dirs(base_dir: Path, prefix: str) -> Iterator[Path]:
    try:
        resolved = base_dir.resolve()
    except OSError:
        resolved = base_dir
    if not resolved.exists():
        return
    if resolved.is_dir() and resolved.name.startswith(prefix):
        yield resolved
    if not resolved.is_dir():
        return
    try:
        entries = list(resolved.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry.is_dir() and entry.name.startswith(prefix):
            yield entry


def _find_backbone_checkpoint_for_config(cfg: YAMLConfig, backbone_root: str | None,
                                         config_path: str) -> Path | None:
    compressed = cfg.yaml_cfg.get('CompressedPResNet')
    if not isinstance(compressed, dict):
        return None
    if compressed.get('compressed_pretrained'):
        return None

    variant = compressed.get('compression_variant') or compressed.get('compression', {}).get('variant')
    if not isinstance(variant, str):
        return None
    try:
        coeff_luma = int(compressed.get('coeff_count_luma', compressed.get('coeff_count', 64)))
    except (TypeError, ValueError):
        coeff_luma = 64
    try:
        coeff_cb = int(compressed.get('coeff_count_cb', compressed.get('coeff_count_chroma', coeff_luma)))
    except (TypeError, ValueError):
        coeff_cb = coeff_luma
    try:
        coeff_cr = int(compressed.get('coeff_count_cr', compressed.get('coeff_count_chroma', coeff_cb)))
    except (TypeError, ValueError):
        coeff_cr = coeff_cb

    prefix = _format_backbone_prefix(variant, coeff_luma, coeff_cb, coeff_cr)

    depth = compressed.get('depth')
    search_dirs: list[Path] = []
    seen: set[Path] = set()

    explicit_root = backbone_root or os.environ.get('COMPRESSED_BACKBONE_ROOT')
    if explicit_root:
        search_dirs.append(Path(explicit_root).expanduser())
    else:
        base_output = Path("output")
        if isinstance(depth, int):
            search_dirs.append(base_output / f"compressed_resnet{depth}")
            search_dirs.append(base_output / f"imagenet_resnet{depth}_backbone")
        search_dirs.append(base_output)

    best_path: Path | None = None
    best_mtime: float | None = None

    for base in search_dirs:
        try:
            resolved = base.resolve()
        except OSError:
            resolved = base
        if resolved in seen:
            continue
        seen.add(resolved)
        for run_dir in _iter_matching_backbone_dirs(resolved, prefix):
            checkpoint = _pick_best_checkpoint_in_dir(run_dir)
            if checkpoint is None:
                continue
            try:
                mtime = checkpoint.stat().st_mtime
            except OSError:
                continue
            if best_mtime is None or mtime > best_mtime:
                best_path = checkpoint
                best_mtime = mtime

    if best_path is None:
        return None

    print(
        "[auto-backbone] Selected checkpoint",
        best_path,
        "from directory",
        best_path.parent,
        "for config",
        config_path,
    )
    return best_path


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    auto_resume_flag = bool(args.auto_resume) if args.auto_resume is not None else False
    checkpoint_root = args.checkpoint_root
    auto_select_backbone = bool(args.auto_select_backbone)
    backbone_root = args.backbone_root

    cli_overrides = {
        k: v for k, v in args.__dict__.items()
        if k not in [
            'update',
            'auto_resume',
            'checkpoint_root',
            'auto_select_backbone',
            'backbone_root',
            'weights_root',
            'time_limit_minutes',
            'time_limit_hours',
        ] and v is not None
    }

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(cli_overrides)

    time_limit_seconds: float | None = None
    minutes_arg = args.time_limit_minutes
    hours_arg = args.time_limit_hours

    if minutes_arg is not None and hours_arg is not None:
        print("[time-limit] Both minutes and hours provided; the hours value will take precedence.")

    if hours_arg is not None:
        try:
            hours_value = float(hours_arg)
        except (TypeError, ValueError):
            print(f"[time-limit] Ignoring invalid hour time limit value: {hours_arg}")
        else:
            if hours_value > 0:
                time_limit_seconds = hours_value * 3600
            else:
                print(f"[time-limit] Ignoring non-positive time limit: {hours_value} hours")
    elif minutes_arg is not None:
        try:
            minutes_value = float(minutes_arg)
        except (TypeError, ValueError):
            print(f"[time-limit] Ignoring invalid minute time limit value: {minutes_arg}")
        else:
            if minutes_value > 0:
                time_limit_seconds = minutes_value * 60
            else:
                print(f"[time-limit] Ignoring non-positive time limit: {minutes_value} minutes")

    if time_limit_seconds is not None:
        update_dict['time_limit_seconds'] = int(time_limit_seconds)

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    raw_weights_root = args.weights_root
    use_default_weights_root = False
    if raw_weights_root is None:
        weights_root = _DEFAULT_WEIGHTS_ROOT
        use_default_weights_root = True
    elif isinstance(raw_weights_root, str):
        weights_root = raw_weights_root.strip() or None
    else:
        weights_root = raw_weights_root

    if auto_resume_flag and not cfg.resume:
        resume_candidate = _find_latest_checkpoint_for_config(
            args.config,
            cfg.output_dir,
            checkpoint_root,
        )
        if resume_candidate is not None:
            cfg.resume = str(resume_candidate)
            print(
                "[auto-resume] Using checkpoint",
                resume_candidate,
                "from directory",
                resume_candidate.parent,
            )
        else:
            print("[auto-resume] No matching checkpoint found; starting from scratch.")

    if auto_select_backbone:
        backbone_candidate = _find_backbone_checkpoint_for_config(
            cfg,
            backbone_root,
            args.config,
        )
        if backbone_candidate is not None:
            cfg.yaml_cfg.setdefault('CompressedPResNet', {})['compressed_pretrained'] = str(backbone_candidate)
        else:
            compressed_cfg = cfg.yaml_cfg.get('CompressedPResNet')
            if isinstance(compressed_cfg, dict) and not compressed_cfg.get('compressed_pretrained'):
                print("[auto-backbone] No matching compressed backbone checkpoint found; proceeding without pretrained weights.")

    if cfg.resume:
        resume_parent = Path(cfg.resume).expanduser()
        if resume_parent.is_file():
            resume_parent = resume_parent.parent
        cfg.output_dir = str(resume_parent)
        cfg.yaml_cfg['output_dir'] = cfg.output_dir
        if not cfg.summary_dir:
            summary_path = resume_parent / "summary"
            cfg.summary_dir = str(summary_path)
            cfg.yaml_cfg['summary_dir'] = cfg.summary_dir
        print(f"[output-dir] Resuming run outputs in {resume_parent}")
    else:
        if use_default_weights_root and weights_root:
            print(f"[output-dir] Using default weights root {weights_root}")
        _prepare_run_directory(cfg, args.config, weights_root)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable Weights & Biases logging (requires wandb)')
    parser.add_argument('--wandb-project', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, help='Weights & Biases entity/team')
    parser.add_argument('--wandb-run-name', type=str, help='Weights & Biases run name')
    parser.add_argument('--wandb-tags', nargs='*', help='Optional list of Weights & Biases tags')
    parser.add_argument('--auto-resume', action=argparse.BooleanOptionalAction, default=None,
                        help='Automatically resume from the latest checkpoint that matches the config name.')
    parser.add_argument('--checkpoint-root', type=str, default=None,
                        help='Directory containing checkpoint subfolders; used alongside --auto-resume.')
    parser.add_argument('--auto-select-backbone', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically load the latest matching compressed backbone checkpoint when available.')
    parser.add_argument('--backbone-root', type=str, default=None,
                        help='Root directory containing compressed backbone checkpoints.')
    parser.add_argument('--weights-root', type=str, default=None,
                        help='Optional root directory to store RT-DETR checkpoints (e.g. rt_detr_weights).')
    parser.add_argument('--time-limit-minutes', type=float, default=None,
                        help='Optional wall-clock time limit (in minutes) before training stops.')
    parser.add_argument('--time-limit-hours', type=float, default=None,
                        help='Optional wall-clock time limit (in hours); takes precedence over minutes when both are set.')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

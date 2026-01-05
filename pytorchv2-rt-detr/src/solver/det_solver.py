"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import numbers

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    _COCO_METRIC_NAMES = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    def _log_wandb(self, epoch: int | None, train_stats: dict | None = None,
                   test_stats: dict | None = None, extra: dict | None = None) -> None:
        if self.wandb_run is None or not dist_utils.is_main_process():
            return
        payload: dict[str, float | int] = {}
        if epoch is not None:
            payload["epoch"] = epoch
        if train_stats:
            payload.update(self._flatten_metrics("train", train_stats))
        if test_stats:
            payload.update(self._flatten_metrics("val", test_stats))
        if extra:
            payload.update(extra)
        if payload:
            self.wandb_run.log(payload)

    def _update_wandb_best(self, epoch: int, test_stats: dict) -> None:
        if self.wandb_run is None or not dist_utils.is_main_process():
            return
        bbox_scores = test_stats.get("coco_eval_bbox")
        if isinstance(bbox_scores, (list, tuple)):
            for name, value in zip(self._COCO_METRIC_NAMES, bbox_scores):
                if isinstance(value, numbers.Number):
                    self.wandb_run.summary[f"best/bbox_{name}"] = float(value)
        self.wandb_run.summary["best/epoch"] = epoch

    @classmethod
    def _flatten_metrics(cls, prefix: str, stats: dict) -> dict[str, float | int]:
        flattened: dict[str, float | int] = {}
        for key, value in stats.items():
            cls._accumulate_metric(flattened, f"{prefix}/{key}", value, key_hint=key)
        return flattened

    @classmethod
    def _accumulate_metric(cls, target: dict[str, float | int], base_key: str, value, *, key_hint: str) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, numbers.Number):
            target[base_key] = float(value)
            return
        if isinstance(value, (list, tuple)):
            names = cls._COCO_METRIC_NAMES if key_hint == "coco_eval_bbox" else None
            for idx, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    item = item.item()
                if isinstance(item, numbers.Number):
                    suffix = names[idx] if names and idx < len(names) else str(idx)
                    target[f"{base_key}_{suffix}"] = float(item)
            return
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                cls._accumulate_metric(target, f"{base_key}/{sub_key}", sub_val, key_hint=sub_key)
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')
        if self.wandb_run is not None and dist_utils.is_main_process():
            self.wandb_run.config.update({"n_parameters": n_parameters}, allow_val_change=True)

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        raw_limit = getattr(args, 'time_limit_seconds', None)
        if raw_limit is None:
            yaml_cfg = getattr(args, 'yaml_cfg', None)
            if isinstance(yaml_cfg, dict):
                raw_limit = yaml_cfg.get('time_limit_seconds')
        time_limit_seconds: float | None = None
        if raw_limit is not None:
            try:
                limit_value = float(raw_limit)
            except (TypeError, ValueError):
                print(f"[time-limit] Ignoring invalid time limit value on config: {raw_limit}")
            else:
                if limit_value > 0:
                    time_limit_seconds = limit_value
                    setattr(args, 'time_limit_seconds', time_limit_seconds)
                    if dist_utils.is_main_process():
                        minutes = time_limit_seconds / 60.0
                        print(f"[time-limit] Training will stop after {minutes:.2f} minutes ({int(time_limit_seconds)} seconds).")
                elif limit_value < 0:
                    print(f"[time-limit] Ignoring non-positive time limit from config: {limit_value}")

        limit_reached = False
        
        for epoch in range(start_epcoch, args.epoches):

            if time_limit_seconds is not None:
                elapsed_before_epoch = time.time() - start_time
                if elapsed_before_epoch >= time_limit_seconds:
                    if dist_utils.is_main_process():
                        print(f"[time-limit] Stopping before epoch {epoch}: time budget of {int(time_limit_seconds)} seconds exhausted.")
                    limit_reached = True
                    break

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            time_control = {'limit': time_limit_seconds, 'reached': False} if time_limit_seconds is not None else None

            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                time_control=time_control,
                run_start_time=start_time,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            limit_reached = bool(time_control and time_control.get('reached'))

            test_stats: dict[str, float] = {}
            coco_evaluator = None

            if not limit_reached:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module, 
                    self.criterion, 
                    self.postprocessor, 
                    self.val_dataloader, 
                    self.evaluator, 
                    self.device
                )

                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
                    
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat['epoch'] == epoch and self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

                if test_stats:
                    print(f'best_stat: {best_stat}')

            extra_metrics = {"model/n_parameters": float(n_parameters)}
            if time_limit_seconds is not None:
                extra_metrics["time_limit_seconds"] = float(time_limit_seconds)
                extra_metrics["time_limit_reached"] = float(limit_reached)

            self._log_wandb(epoch, train_stats, test_stats if test_stats else None, extra=extra_metrics)

            if not limit_reached and best_stat.get('epoch') == epoch:
                self._update_wandb_best(epoch, test_stats)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            if time_limit_seconds is not None:
                log_stats['time_limit_seconds'] = float(time_limit_seconds)
                log_stats['time_limit_reached'] = bool(limit_reached)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if limit_reached:
                if dist_utils.is_main_process():
                    print(f"[time-limit] Time budget reached ({int(time_limit_seconds)} seconds). Stopping after epoch {epoch}.")
                break

            if time_limit_seconds is not None:
                elapsed_after_epoch = time.time() - start_time
                if elapsed_after_epoch >= time_limit_seconds:
                    if dist_utils.is_main_process():
                        print(f"[time-limit] Time budget reached ({int(time_limit_seconds)} seconds). Stopping after epoch {epoch}.")
                    limit_reached = True
                    break

        self._finalize_output_dir_with_epoch()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if self.wandb_run is not None and dist_utils.is_main_process():
            self.wandb_run.summary["training/time_seconds"] = total_time
            self.wandb_run.summary["training/epochs_completed"] = self.last_epoch + 1
            if limit_reached and time_limit_seconds is not None:
                self.wandb_run.summary["training/time_limit_reached"] = True
        self._finish_wandb()


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        self._log_wandb(self.last_epoch, test_stats=test_stats)
        if self.wandb_run is not None and dist_utils.is_main_process():
            bbox_scores = test_stats.get("coco_eval_bbox")
            if isinstance(bbox_scores, (list, tuple)):
                for name, value in zip(self._COCO_METRIC_NAMES, bbox_scores):
                    if isinstance(value, numbers.Number):
                        self.wandb_run.summary[f"eval/bbox_{name}"] = float(value)
        self._finish_wandb()
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return

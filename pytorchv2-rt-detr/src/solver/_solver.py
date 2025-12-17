"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import pickle
import sys
import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict
import atexit

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None 
    return m.to(device) 


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 
        self.wandb_run = None

    def _setup(self, ):
        """Avoid instantiating unnecessary classes 
        """
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model
        
        # NOTE (lyuwenyu): must load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(self.model.to(device), sync_bn=cfg.sync_bn, \
            find_unused_parameters=cfg.find_unused_parameters)

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch
        
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

        self._setup_wandb()

    def cleanup(self, ):
        if self.writer:
            atexit.register(self.writer.close)
        self._finish_wandb()

    def train(self, ):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator

        self._maybe_update_wandb_data_info()

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self, ):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator
        
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

        self._maybe_update_wandb_data_info()

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                v.to(device)

    def state_dict(self):
        """state dict, train/eval
        """
        state = {}
        state['date'] = datetime.now().isoformat()
        
        # TODO for resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict() 

        return state


    def load_state_dict(self, state):
        """load state dict, train/eval
        """
        # TODO
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')


    def load_resume_state(self, path: str):
        """load resume
        """
        # for cuda:0 memory
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = self._load_checkpoint(path)

        self.load_state_dict(state)

    
    def load_tuning_state(self, path: str,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = self._load_checkpoint(path)

        module = dist_utils.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    def _load_checkpoint(self, path: str):
        """Load a checkpoint while handling PyTorch 2.6 safe loading defaults.

        PyTorch 2.6 switches ``torch.load`` to ``weights_only=True`` by default
        which breaks checkpoints that pickle helper classes (for example the
        ``TrainConfig`` dataclass used by the standalone backbone trainer).
        This helper retries the load with an allow-listed stub so we can safely
        consume those files when the user wires them into ``--tuning``.
        """

        try:
            return torch.load(path, map_location='cpu')
        except (pickle.UnpicklingError, AttributeError) as exc:
            message = str(exc)
            safe_types = []

            if 'TrainConfig' in message:
                TrainConfigStub = type('TrainConfig', (), {})
                TrainConfigStub.__module__ = '__main__'
                main_module = sys.modules.get('__main__')
                if main_module is not None and not hasattr(main_module, 'TrainConfig'):
                    setattr(main_module, 'TrainConfig', TrainConfigStub)
                safe_types.append(TrainConfigStub)

            if safe_types and hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_types)
                return torch.load(path, map_location='cpu', weights_only=False)

            raise


    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def _setup_wandb(self) -> None:
        if not getattr(self.cfg, 'wandb', False):
            self.wandb_run = None
            return
        if wandb is None:
            raise RuntimeError("Weights & Biases is not installed. Run 'pip install wandb' to enable logging.")
        if not dist_utils.is_main_process():
            self.wandb_run = None
            return

        run_name = self.cfg.wandb_run_name or self.output_dir.name
        project = self.cfg.wandb_project or 'rtdetr-detection'
        wandb_tags = self.cfg.wandb_tags

        wandb_config = {
            "task": self.cfg.task,
            "epoches": self.cfg.epoches,
            "batch_size": self.cfg.batch_size,
            "use_amp": self.cfg.use_amp,
            "use_ema": self.cfg.use_ema,
            "resume": bool(self.cfg.resume),
            "tuning": bool(self.cfg.tuning),
        }
        if self.cfg.output_dir:
            wandb_config["output_dir"] = str(self.cfg.output_dir)
        if self.cfg.device:
            wandb_config["device"] = self.cfg.device

        self.wandb_run = wandb.init(
            project=project,
            entity=self.cfg.wandb_entity or None,
            name=run_name,
            tags=wandb_tags,
            config=wandb_config,
        )

    def _finish_wandb(self) -> None:
        if self.wandb_run is not None:
            if dist_utils.is_main_process():
                self.wandb_run.finish()
            self.wandb_run = None

    def _maybe_update_wandb_data_info(self) -> None:
        if self.wandb_run is None or not dist_utils.is_main_process():
            return
        info = {}
        train_loader = getattr(self, 'train_dataloader', None)
        if train_loader is not None:
            try:
                info["train_samples"] = len(train_loader.dataset)
            except Exception:
                pass
        val_loader = getattr(self, 'val_dataloader', None)
        if val_loader is not None:
            try:
                info["val_samples"] = len(val_loader.dataset)
            except Exception:
                pass
        if info:
            self.wandb_run.config.update(info, allow_val_change=True)


    def fit(self, ):
        raise NotImplementedError('')


    def val(self, ):
        raise NotImplementedError('')

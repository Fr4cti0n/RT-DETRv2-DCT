"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


def _find_divisor(value: int, preferred: int) -> int:
    if preferred <= 0:
        preferred = 1
    if value % preferred == 0:
        return preferred
    for candidate in range(min(preferred, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _apply_pruned_overrides(cfg: dict) -> dict:
    compressed = cfg.get('CompressedPResNet')
    if not isinstance(compressed, dict):
        return cfg
    if compressed.get('compression_variant') != 'luma-fusion-pruned':
        return cfg

    coeff = compressed.get('coeff_window', 8)
    try:
        coeff_val = int(coeff)
    except (TypeError, ValueError):
        coeff_val = 8
    scale = max(coeff_val / 8.0, 1.0 / 8.0)

    def _scale_value(base: int, minimum: int) -> int:
        return max(minimum, int(round(base * scale)))

    # Hybrid encoder adjustments
    encoder_cfg = cfg.setdefault('HybridEncoder', {})
    base_hidden = encoder_cfg.get('hidden_dim', 256)
    enc_hidden = _scale_value(base_hidden, 32)
    encoder_cfg['hidden_dim'] = enc_hidden
    base_ff = encoder_cfg.get('dim_feedforward', 1024)
    encoder_cfg['dim_feedforward'] = max(enc_hidden * 2, _scale_value(base_ff, 64))
    base_heads = encoder_cfg.get('nhead', 8)
    encoder_cfg['nhead'] = max(1, min(4, _find_divisor(enc_hidden, base_heads)))
    base_depth_mult = encoder_cfg.get('depth_mult', 1.0)
    encoder_cfg['depth_mult'] = max(0.25, float(base_depth_mult) * scale)
    base_expansion = encoder_cfg.get('expansion', 1.0)
    encoder_cfg['expansion'] = max(0.25, float(base_expansion) * scale)
    base_enc_layers = encoder_cfg.get('num_encoder_layers', 1)
    encoder_cfg['num_encoder_layers'] = max(1, _scale_value(base_enc_layers, 1))

    # Transformer decoder adjustments
    decoder_cfg = cfg.setdefault('RTDETRTransformerv2', {})
    dec_base_hidden = decoder_cfg.get('hidden_dim', 256)
    dec_hidden = _scale_value(dec_base_hidden, 32)
    decoder_cfg['hidden_dim'] = dec_hidden
    dec_base_ff = decoder_cfg.get('dim_feedforward', 1024)
    decoder_cfg['dim_feedforward'] = max(dec_hidden * 3, _scale_value(dec_base_ff, 96))
    dec_base_heads = decoder_cfg.get('nhead', 8)
    decoder_cfg['nhead'] = max(2, min(4, _find_divisor(dec_hidden, dec_base_heads)))
    base_layers = decoder_cfg.get('num_layers', 6)
    decoder_cfg['num_layers'] = max(1, min(4, _scale_value(base_layers, 1)))
    base_points = decoder_cfg.get('num_points', 4)
    if isinstance(base_points, (list, tuple)):
        decoder_cfg['num_points'] = [max(1, min(3, int(round(p * scale)))) for p in base_points]
    else:
        decoder_cfg['num_points'] = max(1, min(3, _scale_value(base_points, 1)))
    base_denoising = decoder_cfg.get('num_denoising', 100)
    decoder_cfg['num_denoising'] = max(0, _scale_value(base_denoising, 0))
    decoder_cfg['num_queries'] = 200

    feat_channels = decoder_cfg.get('feat_channels')
    if isinstance(feat_channels, list) and feat_channels:
        decoder_cfg['feat_channels'] = [dec_hidden for _ in feat_channels]
    else:
        decoder_cfg['feat_channels'] = [dec_hidden] * 3

    return cfg


def _apply_channel_scale_overrides(cfg: dict) -> dict:
    """Align downstream in_channels with a channel-scaled CompressedPResNet."""
    compressed = cfg.get('CompressedPResNet')
    if not isinstance(compressed, dict):
        return cfg

    try:
        scale = float(compressed.get('channel_scale', 1.0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        scale = 1.0
    if scale == 1.0:
        return cfg

    try:
        depth = int(compressed.get('depth', 34))
    except (TypeError, ValueError):
        depth = 34
    try:
        return_idx = list(compressed.get('return_idx', [1, 2, 3]))
    except Exception:  # pragma: no cover - defensive
        return_idx = [1, 2, 3]

    expansion = 4 if depth >= 50 else 1

    def _scale_channels(val: int) -> int:
        return max(1, int(round(val * scale)))

    base_channels = [64, 128, 256, 512]
    scaled = [_scale_channels(v) for v in base_channels]
    out_channels = [v * expansion for v in scaled]

    in_channels: list[int] = []
    for idx in return_idx:
        if isinstance(idx, int) and 0 <= idx < len(out_channels):
            in_channels.append(out_channels[idx])

    if in_channels:
        encoder_cfg = cfg.setdefault('HybridEncoder', {})
        encoder_cfg['in_channels'] = in_channels

    return cfg

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)
        cfg = _apply_pruned_overrides(cfg)
        cfg = _apply_channel_scale_overrides(cfg)

        self.yaml_cfg = copy.deepcopy(cfg) 
        
        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)
    
    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model 

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion
    
    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return super().optimizer
    
    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler
    
    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader
    
    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema
    
    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)                
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters() 

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs 

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        print(f'building {name} with batch_size={bs}...')
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)      
        return loader
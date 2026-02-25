from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import GMFCfg, MLPCfg, NeuMFCfg


@dataclass
class Config:
    model: Union[GMFCfg, MLPCfg, NeuMFCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["gmf", "mlp", "neumf"]
    dataset: str
    seed: int
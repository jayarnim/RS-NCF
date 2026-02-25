from dataclasses import dataclass


@dataclass
class GMFCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class MLPCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class NeuMFCfg:
    gmf: GMFCfg
    mlp: MLPCfg
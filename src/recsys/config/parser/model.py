from ..config.model import (
    GMFCfg,
    MLPCfg,
    NeuMFCfg,
)


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="gmf":
        return gmf(cfg)
    elif cls=="mlp":
        return mlp(cfg)
    elif cls=="neumf":
        return neumf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def gmf(cfg):
    return GMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def mlp(cfg):
    return MLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def neumf(cfg):
    gmf = GMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["gmf"],
    )
    mlp = MLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["mlp"],
    )
    return NeuMFCfg(
        gmf=gmf,
        mlp=mlp,
    )
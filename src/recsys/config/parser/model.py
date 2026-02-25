from ..config.model import (
    GMFCfg,
    MLPCfg,
    NeuMFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="gmf":
        return gmf(cfg)
    elif model=="mlp":
        return mlp(cfg)
    elif model=="neumf":
        return neumf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def gmf(cfg):
    return GMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )


def mlp(cfg):
    return MLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def neumf(cfg):
    gmf = GMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["gmf"]["embedding_dim"],
    )
    mlp = MLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["mlp"]["embedding_dim"],
        hidden_dim=cfg["model"]["mlp"]["hidden_dim"],
        dropout=cfg["model"]["mlp"]["dropout"],
    )
    return NeuMFCfg(
        gmf=gmf,
        mlp=mlp,
    )
import torch
import torch.nn as nn
from .components.scorer import LinearProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        gmf: nn.Module,
        mlp: nn.Module,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Neural Matrix Factorization (NeuMF),
        MF, MLP & id embedding based latent factor model,
        combining a Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP)
        to learn low-rank linear representation & high-rank nonlinear user-item extracteds.

        Args:
            gmf (nn.Module)
            mlp (nn.Moudle)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.gmf = gmf
        self.mlp = mlp
        self.predictive_dim = gmf.predictive_dim + mlp.predictive_dim
        
        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        predictive_vec_gmf = self.gmf(user_idx, item_idx)
        predictive_vec_mlp = self.mlp(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(predictive_vec_gmf, predictive_vec_mlp), 
            dim=-1,
        )
        predictive_vec_fusion = torch.cat(**kwargs)

        return predictive_vec_fusion

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            input_dim=self.predictive_dim,
        )
        self.scorer = LinearProjectionLayer(**kwargs)
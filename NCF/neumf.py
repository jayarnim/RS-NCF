import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        gmf: nn.Module,
        ncf: nn.Module,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Neural Matrix Factorization (NeuMF),
        MF, MLP & id embedding based latent factor model,
        combining a Generalized Matrix Factorization (GMF) and a Neural Collaborative Filtering (NCF)
        to learn low-rank linear representation & high-rank nonlinear user-item extracteds.

        Args:
            gmf (nn.Module)
            ncf (nn.Moudle)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.gmf = gmf
        self.ncf = ncf
        self.matching_dim = gmf.matching_dim + ncf.matching_dim
        
        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        matching_vec_gmf = self.gmf(user_idx, item_idx)
        matching_vec_mlp = self.ncf(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(matching_vec_gmf, matching_vec_mlp), 
            dim=-1,
        )
        matching_vec_fusion = torch.cat(**kwargs)

        return matching_vec_fusion

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.matching_dim,
            out_features=1,
        )
        self.prediction = nn.Linear(**kwargs)
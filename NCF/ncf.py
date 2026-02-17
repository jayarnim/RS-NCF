import torch
import torch.nn as nn
from .components.embedding import EmbeddingLayer
from .components.matching import MatchingLayer


class Module(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Neural Collaborative Filtering (NCF),
        MLP & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn high-rank nonlinear user-item extracteds.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_dim (list):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.matching_dim = hidden_dim[-1]

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        matching_vec = self.matching(user_emb, item_emb)
        return matching_vec

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
        pred_vector = self.forward(user_idx, item_idx)
        logit = self.prediction(pred_vector).squeeze(-1)
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
        pred_vector = self.forward(user_idx, item_idx)
        logit = self.prediction(pred_vector).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_components()
        self._create_layers()

    def _create_components(self):
        kwargs = dict(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
        )
        self.embedding = EmbeddingLayer(**kwargs)

        kwargs = dict(
            input_dim=self.embedding_dim*2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.matching = MatchingLayer(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.matching_dim,
            out_features=1,
        )
        self.prediction = nn.Linear(**kwargs)
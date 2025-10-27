import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from models.loss import OutcomeLoss
from tqdm import tqdm


class GroupTokenizer(nn.Module):
    """
    Tokenizer for grouped features:
    - n_continuous features -> 1 token (via MLP)
    - 1 categorical feature -> 1 token (via Embedding)
    - n_outcome features -> n_outcome tokens (each via separate MLP)
    """
    def __init__(self, n_num, categories, n_outcome, embed_dim):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(categories)
        self.categorizes = categories
        self.n_outcome = n_outcome
        self.embed_dim = embed_dim

        # Continuous features encoder (all continuous cols -> 1 embedding)
        if n_num > 0:
            self.continuous_encoder = nn.Sequential(
                nn.Linear(n_num, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )

        # Categorical feature encoder (1 categorical col -> 1 embedding)
        self.categorical_embeddings = nn.ModuleList()
        if len(categories) > 0:
            for n_cat in categories:
                self.categorical_embedding = nn.Embedding(n_cat, embed_dim)
                self.categorical_embeddings.append(self.categorical_embedding)

        # Outcome encoders (each outcome col -> 1 embedding)
        self.outcome_encoders = nn.ModuleList()
        for _ in range(n_outcome):
            self.outcome_encoders.append(nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ))

    def forward(self, x_num=None, x_cat=None, x_outcome=None):
        tokens = []

        # Encode continuous features
        if x_num is not None and self.n_num > 0:
            continuous_token = self.continuous_encoder(x_num)  # [B, embed_dim]
            tokens.append(continuous_token.unsqueeze(1))  # [B, 1, embed_dim]

        # Encode categorical feature
        if x_cat is not None and self.n_cat > 0:
            for i in range(x_cat.shape[1]):
                x_categorical = x_cat[:, i:i+1]  # [B, 1]
                categorical_token = self.categorical_embedding(x_categorical)  # [B, embed_dim]
                tokens.append(categorical_token)  # [B, 1, embed_dim]

        # Encode outcome features
        if x_outcome is not None:
            for i in range(self.n_outcome):
                outcome_i = x_outcome[:, i:i+1]  # [B, 1]
                outcome_token = self.outcome_encoders[i](outcome_i)  # [B, embed_dim]
                tokens.append(outcome_token.unsqueeze(1))  # [B, 1, embed_dim]

        # Concatenate all tokens
        tokens = torch.cat(tokens, dim=1)  # [B, seq_len, embed_dim]
        return tokens


class DIME(nn.Module):
    """
    Conditional generation model for outcomes given input features.

    Architecture:
    - Input: continuous features (grouped), categorical feature, outcome features
    - Tokenization: [continuous_group, categorical, outcome_1, ..., outcome_n]
    - Masking: Only applied to outcome tokens (not input tokens)
    - Goal: Model P(outcomes | continuous, categorical)
    """
    def __init__(
        self,
        n_num,
        categories,
        n_outcome,
        embed_dim=32,
        buffer_size=8,
        depth=6,
        norm_layer=nn.LayerNorm,
        dropout_rate=0.0,
        device='cuda:0'
    ):
        super().__init__()

        self.n_num = n_num
        self.n_cat = len(categories)
        self.n_outcome = n_outcome
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.device = device
        self.attn_drop = dropout_rate
        self.proj_drop = dropout_rate

        # Sequence length: continuous_group (1) + categorical (n_cat) + outcomes (n_outcome)
        # But only outcomes can be masked
        self.n_input_tokens = 0
        if n_num > 0:
            self.n_input_tokens += 1
        if self.n_cat > 0:
            self.n_input_tokens += self.n_cat

        self.seq_len = self.n_input_tokens + n_outcome

        print(f'n_num: {self.n_num}, n_cat: {self.n_cat}, '
              f'n_outcome: {n_outcome}, seq_len: {self.seq_len}')

        # Tokenizer
        self.tokenizer = GroupTokenizer(
            n_num, categories, n_outcome, embed_dim
        )

        # Transformer settings
        num_heads = 4
        mlp_ratio = 16.0

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len + buffer_size, embed_dim)
        )

        # Mask token for masked outcome positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                proj_drop=self.proj_drop,
                attn_drop=self.attn_drop
            ) for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # Outcome loss module (handles all outcomes with masking support)
        self.outcome_loss_module = OutcomeLoss(
            n_outcome=n_outcome,
            hid_dim=embed_dim,
            dim_t=1024,
            dropout_rate=dropout_rate
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Initialize Linear and LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_outcome_masks(self, B):
        """
        Sample random masks for outcome tokens.
        Mask strategy: randomly mask a subset of outcome tokens.

        Returns:
            masks: [B, n_outcome] boolean tensor (True = masked, False = visible)
        """
        masks = []
        for _ in range(B):
            # Randomly decide how many outcomes to mask (at least 1)
            n_to_mask = np.random.randint(1, self.n_outcome + 1)
            mask = torch.zeros(self.n_outcome)

            # Randomly select which outcomes to mask
            mask_indices = np.random.choice(self.n_outcome, n_to_mask, replace=False)
            mask[mask_indices] = 1

            masks.append(mask)

        masks = torch.stack(masks).bool().to(self.device)
        return masks

    def apply_mask_to_tokens(self, tokens, outcome_mask):

        B, seq_len, embed_dim = tokens.shape
        masked_tokens = tokens.clone()

        # Apply mask to outcome tokens (last n_outcome positions)
        outcome_start_idx = self.n_input_tokens
        for i in range(self.n_outcome):
            token_idx = outcome_start_idx + i
            mask_i = outcome_mask[:, i]  # [B]

            # Replace masked positions with mask token
            masked_tokens[mask_i, token_idx, :] = self.mask_token

        return masked_tokens

    def masked_transformer(self, tokens, outcome_mask):

        B, seq_len, embed_dim = tokens.shape

        # Apply masking
        x = self.apply_mask_to_tokens(tokens, outcome_mask)

        # Add buffer tokens (optional, for consistency with original DIME)
        if self.buffer_size > 0:
            buffer_tokens = torch.zeros(B, self.buffer_size, embed_dim, device=self.device)
            x = torch.cat([buffer_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Remove buffer tokens
        if self.buffer_size > 0:
            x = x[:, self.buffer_size:]

        return x

    def forward(self, x_num=None, x_cat=None, x_outcome=None):

        B = x_outcome.shape[0] if x_outcome is not None else x_num.shape[0]

        # Save ground truth outcomes
        gt_outcome = x_outcome.clone().detach()

        # Tokenize all features
        tokens = self.tokenizer(x_num, x_cat, x_outcome)

        # Sample random masks for outcomes
        outcome_mask = self.sample_outcome_masks(B)

        # Apply masked transformer
        hidden_states = self.masked_transformer(tokens, outcome_mask)

        # Extract hidden states for outcome tokens
        outcome_start_idx = self.n_input_tokens
        outcome_hidden = hidden_states[:, outcome_start_idx:, :]  # [B, n_outcome, embed_dim]

        # Compute loss using the outcome loss module
        total_loss, loss_per_outcome = self.outcome_loss_module(
            z_outcome=outcome_hidden,
            gt_outcome=gt_outcome,
            mask=outcome_mask
        )

        return total_loss, loss_per_outcome

    def sample_outcomes(
        self,
        x_num=None,
        x_cat=None,
        outcome_mask=None,
        x_outcome_partial=None,
        num_steps=50,
        device='cuda:0'
    ):

        B = x_num.shape[0] if x_num is not None else x_cat.shape[0]

        # If no mask provided, generate all outcomes
        if outcome_mask is None:
            outcome_mask = torch.ones(B, self.n_outcome, dtype=torch.bool, device=device)

        # Initialize outcomes
        if x_outcome_partial is not None:
            sampled_outcomes = x_outcome_partial.clone()
        else:
            sampled_outcomes = torch.zeros(B, self.n_outcome, device=device)

        # Tokenize inputs (with initial outcome estimates)
        tokens = self.tokenizer(x_num, x_cat, sampled_outcomes)

        # Apply masked transformer
        hidden_states = self.masked_transformer(tokens, outcome_mask)

        # Extract outcome hidden states
        outcome_start_idx = self.n_input_tokens
        outcome_hidden = hidden_states[:, outcome_start_idx:, :]  # [B, n_outcome, embed_dim]

        # Sample outcomes using the loss module
        with torch.no_grad():
            # For each outcome that needs to be sampled
            for i in range(self.n_outcome):
                mask_i = outcome_mask[:, i]  # [B]

                if mask_i.sum() == 0:
                    continue  # Skip if no samples need generation for this outcome

                z_i = outcome_hidden[:, i, :]  # [B, embed_dim]

                # Sample using the outcome loss module
                sampled_i = self.outcome_loss_module.sample_single_outcome(
                    z_outcome_i=z_i,
                    outcome_idx=i,
                    num_steps=num_steps,
                    device=device
                )  # [B, 1]

                # Update sampled outcomes (only for masked positions)
                sampled_outcomes[mask_i, i] = sampled_i[mask_i, 0]

        return sampled_outcomes

    def sample_iterative(
        self,
        x_num=None,
        x_cat=None,
        num_steps=50,
        device='cuda:0'
    ):
        """
        Iteratively sample outcomes in random order (similar to original DIME).

        Args:
            x_num: [B, n_continuous] continuous input features
            x_cat: [B] or [B, 1] categorical input feature
            num_steps: number of diffusion sampling steps per outcome

        Returns:
            sampled_outcomes: [B, n_outcome] sampled outcome values
        """
        B = x_num.shape[0] if x_num is not None else x_cat.shape[0]

        # Initialize outcomes with zeros
        sampled_outcomes = torch.zeros(B, self.n_outcome, device=device)

        # Sample generation order for each sample in batch
        orders = []
        for _ in range(B):
            order = np.random.permutation(self.n_outcome)
            orders.append(order)
        orders = np.array(orders)  # [B, n_outcome]

        # Iteratively generate each outcome
        for step in tqdm(range(self.n_outcome), desc="Sampling outcomes"):
            # Current mask: outcomes not yet generated
            outcome_mask = torch.ones(B, self.n_outcome, dtype=torch.bool, device=device)
            for b in range(B):
                # Unmask already generated outcomes
                for s in range(step):
                    outcome_idx = orders[b, s]
                    outcome_mask[b, outcome_idx] = False

            # Tokenize with current partial outcomes
            tokens = self.tokenizer(x_num, x_cat, sampled_outcomes)

            # Apply masked transformer
            hidden_states = self.masked_transformer(tokens, outcome_mask)

            # Extract outcome hidden states
            outcome_start_idx = self.n_input_tokens
            outcome_hidden = hidden_states[:, outcome_start_idx:, :]  # [B, n_outcome, embed_dim]

            # Generate current outcome for each sample
            with torch.no_grad():
                for b in range(B):
                    outcome_idx = orders[b, step]
                    z_i = outcome_hidden[b, outcome_idx, :]  # [embed_dim]

                    # Sample using the outcome loss module
                    sampled_i = self.outcome_loss_module.sample_single_outcome(
                        z_outcome_i=z_i.unsqueeze(0),
                        outcome_idx=outcome_idx,
                        num_steps=num_steps,
                        device=device
                    )  # [1, 1]

                    sampled_outcomes[b, outcome_idx] = sampled_i[0, 0]

        return sampled_outcomes

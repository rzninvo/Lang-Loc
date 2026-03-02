"""
Wrapper for DualSceneAligner that adds matching probability prediction.
FIXED to handle contrastive training checkpoint format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualSceneAlignerWithMatching(nn.Module):
    """
    Wrapper that adds matching probability prediction to DualSceneAligner.
    """
    
    def __init__(self, base_model, hidden_dim=256, use_cosine=True):
        """
        Args:
            base_model: Your trained DualSceneAligner model
            hidden_dim: Dimension of embeddings from base_model (256 for your model)
            use_cosine: If True, use cosine similarity (recommended)
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.use_cosine = use_cosine
        
        if not use_cosine:
            # MLP matching head
            self.matching_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, batch):
        """
        Returns:
            dict with "src_emb", "ref_emb", "matching_prob"
        """
        # Get embeddings from base model
        out = self.base_model(batch)
        src_emb = out["src_emb"]
        ref_emb = out["ref_emb"]
        
        if self.use_cosine:
            # Use cosine similarity as matching probability
            cos_sim = F.cosine_similarity(src_emb, ref_emb, dim=-1)
            matching_prob = (cos_sim + 1) / 2  # Map [-1, 1] to [0, 1]
        else:
            # Use MLP
            concat_emb = torch.cat([src_emb, ref_emb], dim=-1)
            matching_prob = self.matching_head(concat_emb).squeeze(-1)
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb,
            "matching_prob": matching_prob
        }


def load_model_with_matching(checkpoint_path, base_model_config, 
                             hidden_dim=256, use_cosine=True, device='cpu'):
    """
    Load a trained DualSceneAligner and wrap it with matching head.
    FIXED to handle contrastive training checkpoint format!
    """
    from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner    
    # Create base model
    base_model = DualSceneAligner(**base_model_config).to(device)
    
    # Load trained weights - HANDLE DIFFERENT FORMATS
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Contrastive training format (your current checkpoints)
        print("✓ Detected contrastive training checkpoint format")
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training stats if available
        if 'separation' in checkpoint:
            print(f"  Separation: {checkpoint['separation']:.4f}")
            print(f"  Positive sim: {checkpoint.get('pos_sim', 0):.4f}")
            print(f"  Negative sim: {checkpoint.get('neg_sim', 0):.4f}")
    elif 'model' in checkpoint:
        # Old format
        print("✓ Detected old checkpoint format")
        base_model.load_state_dict(checkpoint['model'])
    else:
        # Direct state dict
        print("✓ Detected direct state dict format")
        base_model.load_state_dict(checkpoint)
    
    # Wrap with matching head
    model = DualSceneAlignerWithMatching(
        base_model=base_model,
        hidden_dim=hidden_dim,
        use_cosine=use_cosine
    ).to(device)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    if not use_cosine:
        print(f"  Matching head parameters: {sum(p.numel() for p in model.matching_head.parameters()):,}")
    else:
        print(f"  Using cosine similarity (no extra parameters)")
    
    return model
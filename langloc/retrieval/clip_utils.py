import torch
import clip

_device = "cpu"
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)

@torch.no_grad()
def get_clip_text_embedding(text: str):
    """
    Returns a 512-d CLIP text embedding.
    """
    tokens = clip.tokenize([text]).to(_device)
    emb = _clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]  # (512,)
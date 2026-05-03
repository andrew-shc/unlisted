"""
- GT Image preprocessing (e.g., masking)
- View generation utils on a random sphere or just Fibonacci sphere
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()


def clip_text_similarity(view_of_predictions: list[Image], text: str):
    inputs = processor(text=text, images=view_of_predictions, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embeds = model.get_text_features(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        image_embeds = model.get_image_features(**{k: v for k, v in inputs.items() if k == "pixel_values"})
    
    text_embeds = F.normalize(text_embeds, dim=-1)  # .pooler_output
    image_embeds = F.normalize(image_embeds, dim=-1)  # .pooler_output
    
    # shape: (n images, 1 text)
    similarity = image_embeds @ text_embeds.T
    avg = similarity[:,0].mean().item()
    return avg


def clip_img_similarity(view_of_gts: list[Image], observered_render: Image, view_of_predictions: list[Image]):
    ref_inputs = {k: v.to(device) for k, v in processor(images=view_of_gts + [observered_render], return_tensors="pt", padding=True).items()}
    pred_inputs = {k: v.to(device) for k, v in processor(images=view_of_predictions, return_tensors="pt", padding=True).items()}

    with torch.no_grad():
        reference_embeds = model.get_image_features(**ref_inputs)
        pred_embeds = model.get_image_features(**pred_inputs)

    reference_embeds = F.normalize(reference_embeds, dim=-1)  # .pooler_output
    pred_embeds = F.normalize(pred_embeds, dim=-1)  # .pooler_output

    avg_reference = reference_embeds.mean(dim=0, keepdim=True)
    avg_reference = F.normalize(avg_reference, dim=-1)

    # shape: (n predictions, 1 reference)
    similarity = pred_embeds @ avg_reference.T
    avg = similarity[:, 0].mean().item()
    return avg






def load_images_from_dir(folder: Path) -> list[Image.Image]:
    imgs = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            imgs.append(Image.open(f).convert("RGBA"))
    return imgs

def get_pred_images(pred_dir: Path, model: str) -> list[Image.Image] | None:
    model_dir = pred_dir / model
    if not model_dir.exists():
        return None
    if model == "SAM3D":
        relaxflow = model_dir / "relaxflow" / "frames_rgba"
        if not relaxflow.exists():
            return None
        imgs = load_images_from_dir(relaxflow)
    else:
        imgs = load_images_from_dir(model_dir)
    return imgs if imgs else None

def get_pred_text(pred_dir: Path) -> str | None:
    """Filename (sans extension) of the single image directly inside predX/."""
    for f in pred_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return f.stem
    return None

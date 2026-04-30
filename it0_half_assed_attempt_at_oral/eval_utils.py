"""
- GT Image preprocessing (e.g., masking)
- View generation utils on a random sphere or just Fibonacci sphere
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()


def clip_text_similarity(view_of_predictions: list[Image], text: str):
    inputs = processor(text=text, images=view_of_predictions, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        image_embeds = model.get_image_features(**{k: v for k, v in inputs.items() if k == "pixel_values"})
    
    text_embeds = F.normalize(text_embeds.pooler_output, dim=-1)
    image_embeds = F.normalize(image_embeds.pooler_output, dim=-1)
    
    # shape: (n images, 1 text)
    similarity = image_embeds @ text_embeds.T
    avg = similarity[:,0].mean().item()
    return avg


def clip_img_similarity(view_of_gts: list[Image], observered_render: Image, view_of_predictions: list[Image]):
    ref_inputs = processor(images=view_of_gts + [observered_render], return_tensors="pt", padding=True)
    pred_inputs = processor(images=view_of_predictions, return_tensors="pt", padding=True)

    with torch.no_grad():
        reference_embeds = model.get_image_features(**ref_inputs)
        pred_embeds = model.get_image_features(**pred_inputs)

    reference_embeds = F.normalize(reference_embeds.pooler_output, dim=-1)
    pred_embeds = F.normalize(pred_embeds.pooler_output, dim=-1)

    avg_reference = reference_embeds.mean(dim=0, keepdim=True)
    avg_reference = F.normalize(avg_reference, dim=-1)

    # shape: (n predictions, 1 reference)
    similarity = pred_embeds @ avg_reference.T
    avg = similarity[:, 0].mean().item()
    return avg



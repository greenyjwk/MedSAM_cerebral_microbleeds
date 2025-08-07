import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from skimage import transform
import sys

# Add MedSAM directory to path
sys.path.append('/media/Datacenter_storage/Ji/MedSAM')

def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-6)

def compute_iou(pred, gt):
    return jaccard_score(gt.flatten(), pred.flatten())

def extract_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def load_medsam_model(checkpoint_path, device='cuda:0'):
    """Load the fine-tuned MedSAM model"""
    try:
        from segment_anything import sam_model_registry
        from train_one_gpu import MedSAM
        
        # Load base SAM model
        sam_checkpoint = "/media/Datacenter_storage/Ji/MedSAM/finetuned_weights/MedSAM-ViT-B-20250805-1445/medsam_model_best.pth"
        sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        # sam_model = sam_model_registry["vit_b"]()
        medsam_model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(device)
        
        # Load fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        medsam_model.load_state_dict(checkpoint["model"])
        medsam_model.eval()
        
        print(f"✅ Loaded MedSAM model from epoch {checkpoint.get('epoch', 'unknown')}")
        return medsam_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def inference_medsam(model, image_path, bbox, device='cuda:0'):
    """Run inference with MedSAM model"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Resize image to 1024x1024 (MedSAM input size)
        image_1024 = transform.resize(
            image_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        
        # Convert to tensor (H,W,C) -> (C,H,W) and normalize
        image_tensor = torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        
        # Scale bbox coordinates to 1024x1024
        h_scale = 1024 / image_np.shape[0]
        w_scale = 1024 / image_np.shape[1]
        
        scaled_bbox = [
            int(bbox[0] * w_scale),  # x_min
            int(bbox[1] * h_scale),  # y_min  
            int(bbox[2] * w_scale),  # x_max
            int(bbox[3] * h_scale)   # y_max
        ]
        
        bbox_tensor = torch.tensor(scaled_bbox).float().unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            pred_mask = model(image_tensor, bbox_tensor)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().numpy().squeeze()
        
        # Convert to binary mask and resize back to original size
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
        pred_mask_resized = transform.resize(
            pred_mask_binary, image_np.shape[:2], order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        
        return pred_mask_resized > 0
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return None

# === 🔧 PATHS ===
image_dir = "/media/Datacenter_storage/Ji/valdo_dataset/valdo_t2s_cmbOnly/images/val"
mask_dir = "/media/Datacenter_storage/Ji/valdo_dataset/valdo_t2s_cmbOnly/masks/val"
checkpoint_path = "/media/Datacenter_storage/Ji/MedSAM/finetuned_weights/MedSAM-ViT-B-20250805-1445/medsam_model_best.pth"
output_dir = "/media/Datacenter_storage/Ji/MedSAM/overall_eval"
os.makedirs(output_dir, exist_ok=True)

# === DEVICE SETUP ===
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# === LOAD MODEL ===
print("Loading MedSAM model...")
medsam_model = load_medsam_model(checkpoint_path, device)
if medsam_model is None:
    print("❌ Failed to load model. Exiting.")
    exit(1)

# === 🖼️ GET ALL IMAGE AND MASK FILES ===
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")

# === 📊 METRICS LISTS ===
dice_scores = []
iou_scores = []

# === 🚀 LOOP THROUGH DATA ===
for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
    img_name = os.path.basename(img_path)
    out_mask_path = os.path.join(output_dir, img_name)

    # Load ground truth mask
    gt_mask = np.array(Image.open(mask_path).convert("L")) > 128
    bbox = extract_bbox(gt_mask)
    if bbox is None:
        continue

    # Run MedSAM inference
    pred_mask = inference_medsam(medsam_model, img_path, bbox, device)
    if pred_mask is None:
        continue

    # Save predicted mask
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(out_mask_path)

    # Compute metrics
    dice = compute_dice(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)

    dice_scores.append(dice)
    iou_scores.append(iou)

# === ✅ FINAL REPORT ===
if dice_scores and iou_scores:
    print("\n✅ Evaluation Complete")
    print(f"Average Dice Score over {len(dice_scores)} images: {np.mean(dice_scores):.4f}")
    print(f"Average IoU Score  over {len(iou_scores)} images: {np.mean(iou_scores):.4f}")
else:
    print("\n❌ No successful predictions made")
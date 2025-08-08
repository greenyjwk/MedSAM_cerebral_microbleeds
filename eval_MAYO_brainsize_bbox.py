import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
# Removed sklearn import since no metrics computation needed
from skimage import transform
import sys

# Add MedSAM directory to path
sys.path.append('/media/Datacenter_storage/Ji/MedSAM')

# Removed compute_dice and compute_iou functions since no ground truth available

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Convert YOLO format (x_center, y_center, width, height) to bbox format (x_min, y_min, x_max, y_max)
    YOLO coordinates are normalized (0-1), bbox coordinates are in pixels
    
    Args:
        yolo_coords: [x_center, y_center, width, height] in normalized format
        img_width: image width in pixels
        img_height: image height in pixels
    
    Returns:
        bbox: [x_min, y_min, x_max, y_max] in pixel coordinates
    """
    x_center, y_center, width, height = yolo_coords
    
    # Convert from normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate bbox corners
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return [x_min, y_min, x_max, y_max]

def load_yolo_bbox(txt_path, img_width, img_height):
    """
    Load YOLO format bounding box from txt file
    
    Args:
        txt_path: path to YOLO format txt file
        img_width: image width in pixels
        img_height: image height in pixels
    
    Returns:
        bbox: [x_min, y_min, x_max, y_max] in pixel coordinates, or None if file doesn't exist or is empty
    """
    if not os.path.exists(txt_path):
        print(f"Warning: YOLO file not found: {txt_path}")
        return None
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: Empty YOLO file: {txt_path}")
            return None
        
        # Take the first bounding box (assuming single object per image)
        # YOLO format: class_id x_center y_center width height
        line = lines[0].strip().split()
        if len(line) < 5:
            print(f"Warning: Invalid YOLO format in {txt_path}")
            return None
        
        # Extract YOLO coordinates (skip class_id)
        yolo_coords = [float(x) for x in line[1:5]]  # [x_center, y_center, width, height]
        
        # Convert to bbox format
        bbox = yolo_to_bbox(yolo_coords, img_width, img_height)
        return bbox
        
    except Exception as e:
        print(f"Error reading YOLO file {txt_path}: {e}")
        return None

def load_medsam_model(checkpoint_path, device='cuda:0'):
    """Load the fine-tuned MedSAM model"""
    try:
        from segment_anything import sam_model_registry
        from train_one_gpu import MedSAM
        
        # Load base SAM model
        sam_checkpoint = "/media/Datacenter_storage/Ji/MedSAM/finetuned_weights/MedSAM-ViT-B-20250806-0753/medsam_model_best.pth"
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


def get_bounding_box(mask_tensor):
    print(mask_tensor.shape)
    # sys.exit()

    mask_tensor = torch.squeeze(mask_tensor)
    mask_tensor = mask_tensor.sum(dim=0)
    mask_tensor = (mask_tensor > 0).int()

    nonzero = torch.nonzero(mask_tensor, as_tuple=False)

    h_top = torch.min(nonzero[:,0]).item()
    h_bottom = torch.max(nonzero[:,0]).item()
    w_left = torch.min(nonzero[:,1]).item()
    w_right = torch.max(nonzero[:,1]).item()
    return [w_left, h_top, w_right, h_bottom]


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
        
        # # Scale bbox coordinates to 1024x1024
        # h_scale = 1024 / image_np.shape[0]
        # w_scale = 1024 / image_np.shape[1]
        
        # scaled_bbox = [
        #     int(bbox[0] * w_scale),  # x_min
        #     int(bbox[1] * h_scale),  # y_min  
        #     int(bbox[2] * w_scale),  # x_max
        #     int(bbox[3] * h_scale)   # y_max
        # ]
        
        # bbox_tensor = torch.tensor(scaled_bbox).float().unsqueeze(0).to(device)
        # bbox_tensor = torch.tensor([0, 0, 1024, 1024], device='cuda:0', dtype=torch.float32).unsqueeze(0)
      
        bbox_tensor = get_bounding_box(image_tensor)
        bbox_tensor = torch.tensor([bbox_tensor[0], bbox_tensor[1], bbox_tensor[2], bbox_tensor[3]], device = 'cuda:0', dtype=torch.float32).unsqueeze(0)
        print(bbox)
        print()
        # sys.exit()

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
image_dir = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_t2s_png/images/test"
yolo_dir = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_t2s_png/labels/test"  # Path to YOLO txt files
checkpoint_path = "/media/Datacenter_storage/Ji/MedSAM/finetuned_weights/MedSAM-ViT-B-20250806-0753/medsam_model_best.pth"
output_dir = "/media/Datacenter_storage/Ji/MedSAM/TEMP"
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

# === 🖼️ GET ALL IMAGE FILES ===
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
yolo_paths = sorted(glob(os.path.join(yolo_dir, "*.txt")))

print(f"Found {len(image_paths)} images and {len(yolo_paths)} YOLO files")

# === 📊 PROCESSING COUNTERS ===
processed_count = 0
skipped_count = 0

# === 🚀 LOOP THROUGH DATA ===
for img_path in tqdm(image_paths, desc="Processing images"):
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    
    # Construct corresponding paths
    yolo_path = os.path.join(yolo_dir, base_name + ".txt")
    out_mask_path = os.path.join(output_dir, img_name)
    
    # Load image to get dimensions
    try:
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
        skipped_count += 1
        continue
    
    # Load YOLO bounding box
    bbox = load_yolo_bbox(yolo_path, img_width, img_height)
    # if bbox is None:
    #     print(f"Warning: Could not load YOLO bbox for {img_name}")
    #     skipped_count += 1
    #     continue
    
    # Run MedSAM inference
    pred_mask = inference_medsam(medsam_model, img_path, bbox, device)
    if pred_mask is None:
        print(f"Warning: Inference failed for {img_name}")
        skipped_count += 1
        continue
    
    # Save predicted mask
    try:
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(out_mask_path)
        processed_count += 1
        print(f"✅ Processed {img_name} - bbox: {bbox}")
    except Exception as e:
        print(f"Error saving predicted mask for {img_name}: {e}")
        skipped_count += 1

# === ✅ FINAL REPORT ===
print(f"\n✅ Processing Complete")
print(f"Total images: {len(image_paths)}")
print(f"Successfully processed: {processed_count}")
print(f"Skipped: {skipped_count}")
print(f"Predictions saved to: {output_dir}")

if processed_count > 0:
    print(f"\n🎯 Successfully generated {processed_count} mask predictions using YOLO bounding boxes!")
else:
    print("\n❌ No successful predictions made")
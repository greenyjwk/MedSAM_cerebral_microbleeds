import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage import transform

TRUE_POSITVE = 0
FALSE_NEGATIVE = 0
FALSE_POSITIVE = 0
SEGMENTATION_THRESHOLD = 1.0

# Set GPU device
gpu = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
torch.cuda.set_device(0)

def get_bounding_box(mask_tensor):
    print(mask_tensor.shape)

    mask_tensor = torch.squeeze(mask_tensor)
    mask_tensor = mask_tensor.sum(dim=0)
    mask_tensor = (mask_tensor > 0).int()
    nonzero = torch.nonzero(mask_tensor, as_tuple=False)

    h_top = torch.min(nonzero[:,0]).item()
    h_bottom = torch.max(nonzero[:,0]).item()
    w_left = torch.min(nonzero[:,1]).item()
    w_right = torch.max(nonzero[:,1]).item()
    return [w_left, h_top, w_right, h_bottom]


def inference_medsam(model, image_path, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Resize image to 1024x1024 (MedSAM input size)
        image_1024 = transform.resize(
            image_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        
        # Convert to tensor (H,W,C) -> (C,H,W) and normalize
        image_tensor = torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        
        # Tight brain size prompt bounding box
        bbox_tensor = get_bounding_box(image_tensor)
        bbox_tensor = torch.tensor([bbox_tensor[0], bbox_tensor[1], bbox_tensor[2], bbox_tensor[3]], device = 'cuda:0', dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(image_tensor, bbox_tensor)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().numpy().squeeze()

        # Convert to binary mask and resize back to original size
        pred_mask_binary = (pred_mask > SEGMENTATION_THRESHOLD).astype(np.uint8)
        pred_mask_resized = transform.resize(
            pred_mask_binary, image_np.shape[:2], order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        
        return pred_mask_resized > 0
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return None

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
        medsam_model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(device)

        # Load fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        medsam_model.load_state_dict(checkpoint["model"])
        medsam_model.eval()
        return medsam_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def inference(file_path, checkpoint_path, device):
    medsam_model = load_medsam_model(checkpoint_path, device)    
    pred_masks = inference_medsam(medsam_model, file_path, device='cuda:0')
    pred_mask, num_clusters = label(pred_masks)
    return pred_mask, num_clusters

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def yolo_gt_seg_pred_overlap_check(gt_path, pred_mask):
    global TRUE_POSITVE, FALSE_NEGATIVE, FALSE_POSITIVE
    gt_box_list = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                gt_box_list.append([float(x) for x in parts[1:]])  # YOLO format

    print(pred_mask)
    width = pred_mask.shape[0]
    num_features = pred_mask.max()
    matched_preds = set()
    matched_gts = set()

    for gt_idx, gt_box in enumerate(gt_box_list):
        x_center, y_center, box_w, box_h = [gt_box[i] * width for i in range(4)]
        x1 = int(x_center - box_w / 2)
        x2 = int(x_center + box_w / 2)
        y1 = int(y_center - box_h / 2)
        y2 = int(y_center + box_h / 2)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, width)

        gt_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        gt_mask[y1:y2, x1:x2] = 1

        found_match = False
        for i in range(num_features):
            if i in matched_preds:
                continue
            pred_segment = (pred_mask == (i + 1)).astype(np.uint8)
            iou = compute_iou(gt_mask, pred_segment)
            if iou > 0.01:  # threshold for match
                TRUE_POSITVE += 1
                matched_preds.add(i)
                matched_gts.add(gt_idx)
                found_match = True
                break
        if not found_match:
            FALSE_NEGATIVE += 1
    FALSE_POSITIVE += (num_features - len(matched_preds))

def main():
    img_root_path = "/media/Datacenter_storage/Ji/valdo_dataset/valdo_t2s_cmbOnly/images/val"
    gt_root_path = "/media/Datacenter_storage/Ji/valdo_dataset/valdo_t2s_cmbOnly/labels/val"
    checkpoint_path = "/media/Datacenter_storage/Ji/MedSAM/finetuned_weights/MedSAM-ViT-B-20250806-0753/medsam_model_best.pth"

    cnt = 0
    for img_path in os.listdir(img_root_path):
        img_full_path = os.path.join(img_root_path, img_path)
        gt_path = img_path.replace("png", "txt")
        full_gt_path = os.path.join(gt_root_path, gt_path)
        print(img_full_path)
        pred_mask, num_clusters = inference(img_full_path, checkpoint_path, device='cuda:0')
        yolo_gt_seg_pred_overlap_check(full_gt_path, pred_mask)

        cnt += 1
        if cnt % 2  == 0:
        # if cnt % 50  == 0:
            print("TRUE_POSTIVE", TRUE_POSITVE)
            print("FALSE_NEGATIVE", FALSE_NEGATIVE)
            print("FALSE_POSITIVE", FALSE_POSITIVE)
            print("Image Processed:", cnt)
            print()

if __name__ == "__main__":
    main()

print("TRUE_POSITVE:", TRUE_POSITVE)
print("FALSE_NEGATIVE:", FALSE_NEGATIVE)
print("FALSE_POSITIVE:", FALSE_POSITIVE)
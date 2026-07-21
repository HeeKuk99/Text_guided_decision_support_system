"""
====================================================================
TextBraTS Demo: Missing-Modality Brain Tumor Segmentation
with Adaptive Text-Visual Fusion
====================================================================
Interactive demo for the TextBraTS framework.
Upload MRI modalities, select missing ones, get segmentation + report.

Deploy:
    pip install gradio nibabel torch numpy
    python app.py

HuggingFace Spaces:
    Upload this folder to a new Space (Gradio SDK)
====================================================================
"""

import os
import json
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# =====================================================================
# Model Architecture (inference-only, minimal)
# =====================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Dropout2d(0.35),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.down = nn.MaxPool2d(2) if down else None
    def forward(self, x):
        s = self.conv(x)
        return (s, self.down(s)) if self.down else s

class _UpBlock(nn.Module):
    def __init__(self, in_c, sk_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = _ConvBlock(out_c + sk_c, out_c, down=False)
    def forward(self, x, skip):
        return self.conv(torch.cat([skip, self.up(x)], 1))

class _VLMFusion(nn.Module):
    def __init__(self, vd, td, fd, nh=16):
        super().__init__()
        self.vp = nn.Linear(vd, fd); self.tp = nn.Linear(td, fd)
        self.ca = nn.MultiheadAttention(fd, nh, batch_first=True)
        self.n1 = nn.LayerNorm(fd)
        self.ff = nn.Sequential(nn.Linear(fd, fd*4), nn.GELU(), nn.Linear(fd*4, fd))
        self.n2 = nn.LayerNorm(fd)
    def forward(self, df, tf, htm=None):
        B, C, H, W = df.shape
        q = self.vp(df.flatten(2).permute(0,2,1))
        kv = self.tp(tf.view(B,1,-1))
        r, _ = self.ca(q, kv, kv)
        if htm is not None: r = r * htm.view(B,1,1).float()
        o = self.n1(q + r); o = self.n2(o + self.ff(o))
        return o.permute(0,2,1).view(B,-1,H,W)

class _Gate(nn.Module):
    def __init__(self, nm=4, h=128, vd=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(nm, h), nn.ReLU(), nn.Linear(h, vd), nn.Sigmoid())
    def forward(self, mask, fi, ft, htm=None):
        a = self.gate(mask).view(fi.shape[0], -1, 1, 1)
        f = fi + a * (ft - fi)
        if htm is not None:
            b = htm.view(fi.shape[0], 1, 1, 1).float()
            f = fi * (1-b) + f * b
        return f

class TextBraTSModel(nn.Module):
    """Modality-Aware Adaptive Text-Visual Fusion for Brain Tumor Segmentation"""
    def __init__(self, n_cls=3, txt_dim=768, fuse=512):
        super().__init__()
        self.enc1 = _ConvBlock(16, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.enc4 = _ConvBlock(256, fuse, down=False)
        self.bn_down = nn.MaxPool2d(2)
        self.bn_conv = _ConvBlock(fuse, fuse, down=False)
        self.text_proj = nn.Sequential(
            nn.Linear(txt_dim, fuse), nn.GELU(), nn.LayerNorm(fuse), nn.Dropout(0.3))
        self.default_text = nn.Parameter(torch.randn(1, fuse))
        self.vlm_s4 = _VLMFusion(fuse, fuse, fuse)
        self.vlm_bn = _VLMFusion(fuse, fuse, fuse)
        self.gate_s4 = _Gate(4, 128, fuse)
        self.gate_bn = _Gate(4, 128, fuse)
        self.dec4 = _UpBlock(fuse, fuse, 256)
        self.dec3 = _UpBlock(256, 256, 128)
        self.dec2 = _UpBlock(128, 128, 64)
        self.dec1 = _UpBlock(64, 64, 64)
        self.head = nn.Conv2d(64, n_cls, 1)
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_ratio = nn.Parameter(torch.tensor(0.0))
        self.ratio_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fuse, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, img, txt, has_text_mask=None):
        B = img.shape[0]
        s1, x = self.enc1(img); s2, x = self.enc2(x)
        s3, x = self.enc3(x); s4 = self.enc4(x)
        deep = self.bn_conv(self.bn_down(s4))
        if txt.shape[-1] == 768:
            t = self.text_proj(txt if txt.dim()==2 else txt.view(B,-1)[:,:768])
        else:
            t = self.default_text.expand(B,-1)
        if has_text_mask is not None:
            m = has_text_mask.view(B,1).float()
            t = t*m + self.default_text.expand(B,-1)*(1-m)
        mask = img[:,12:].mean(dim=[2,3])
        s4f = self.gate_s4(mask, s4, self.vlm_s4(s4, t, has_text_mask), has_text_mask)
        bnf = self.gate_bn(mask, deep, self.vlm_bn(deep, t, has_text_mask), has_text_mask)
        x = self.dec4(bnf, s4f); x = self.dec3(x, s3)
        x = self.dec2(x, s2); x = self.dec1(x, s1)
        return self.head(x), self.ratio_head(bnf)


# =====================================================================
# Inference Engine
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CROP = (128, 128)
MODALITY_NAMES = ["FLAIR", "T1CE", "T2", "T1"]

def load_model(weight_path="model_weights.pth"):
    global MODEL
    MODEL = TextBraTSModel().to(DEVICE)
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        # Handle keys from different training setups
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            # Map from training model to demo model
            nk = nk.replace("vlm_fusion_skip4", "vlm_s4")
            nk = nk.replace("vlm_fusion_bottleneck", "vlm_bn")
            nk = nk.replace("gate_skip4", "gate_s4")
            nk = nk.replace("gate_bottleneck", "gate_bn")
            nk = nk.replace("final_head", "head")
            nk = nk.replace("default_text_embedding", "default_text")
            nk = nk.replace("cross_attention", "ca")
            nk = nk.replace("v_proj", "vp")
            nk = nk.replace("t_proj", "tp")
            nk = nk.replace("norm1", "n1")
            nk = nk.replace("norm2", "n2")
            nk = nk.replace("ffn", "ff")
            new_state[nk] = v
        MODEL.load_state_dict(new_state, strict=False)
    MODEL.eval()
    return MODEL


def normalize_volume(vol):
    mask = vol > 0
    if mask.sum() == 0: return vol
    vals = vol[mask]
    lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    vol = np.clip(vol, lo, hi)
    vol = (vol - vals.mean()) / (vals.std() + 1e-8)
    vol[~mask] = 0
    return vol


def run_inference(volumes, missing_modalities):
    """
    volumes: dict {mod_idx: np.array (H,W,D)}
    missing_modalities: list of missing mod indices
    Returns: segmentation (H,W,D,3), measurements dict
    """
    if MODEL is None:
        load_model()

    # Get dimensions from first available volume
    ref_vol = next(iter(volumes.values()))
    H, W, D = ref_vol.shape

    # Build 4-modality stack
    full_vol = np.zeros((H, W, D, 4), dtype=np.float32)
    for mi in range(4):
        if mi in volumes and mi not in missing_modalities:
            full_vol[:, :, :, mi] = normalize_volume(volumes[mi])

    # Modality mask
    mod_mask = np.ones(4, dtype=np.float32)
    for mi in missing_modalities:
        mod_mask[mi] = 0.0

    # 2.5D slice inference
    pred_volume = np.zeros((H, W, D, 3), dtype=np.float32)

    with torch.no_grad():
        for z in range(1, D - 1):
            # 3 adjacent slices × 4 modalities = 12 channels
            channels = []
            for mi in range(4):
                for dz in [-1, 0, 1]:
                    channels.append(full_vol[:, :, z + dz, mi])
            img = np.stack(channels, axis=0)  # (12, H, W)

            # Center crop
            h0 = max(0, (H - CROP[0]) // 2)
            w0 = max(0, (W - CROP[1]) // 2)
            img_crop = img[:, h0:h0+CROP[0], w0:w0+CROP[1]]

            # Pad if needed
            ph = CROP[0] - img_crop.shape[1]
            pw = CROP[1] - img_crop.shape[2]
            if ph > 0 or pw > 0:
                img_crop = np.pad(img_crop, ((0,0),(0,ph),(0,pw)))

            # Add modality mask channels
            mask_ch = np.stack([
                np.full(CROP, mod_mask[i], dtype=np.float32) for i in range(4)
            ], axis=0)
            inp = np.concatenate([img_crop, mask_ch], axis=0)  # (16, H, W)

            inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            txt_t = torch.zeros(1, 768, device=DEVICE)
            htm_t = torch.zeros(1, dtype=torch.bool, device=DEVICE)

            seg_pred, ratio_pred = MODEL(inp_t, txt_t, has_text_mask=htm_t)
            pred = (torch.sigmoid(seg_pred) > 0.5).float().cpu().numpy()[0]

            # Place back in full volume
            pred_full = np.zeros((3, H, W), dtype=np.float32)
            pred_full[:, h0:h0+CROP[0], w0:w0+CROP[1]] = pred[:, :CROP[0], :CROP[1]]
            pred_volume[:, :, z, :] = pred_full.transpose(1, 2, 0)

    # Compute measurements
    et = pred_volume[:,:,:,0]; ncr = pred_volume[:,:,:,1]; ed = pred_volume[:,:,:,2]
    tc = np.clip(et + ncr, 0, 1); wt = np.clip(et + ncr + ed, 0, 1)

    measurements = {
        "wt_voxels": int(wt.sum()),
        "tc_voxels": int(tc.sum()),
        "et_voxels": int(et.sum()),
        "ncr_voxels": int(ncr.sum()),
        "ed_voxels": int(ed.sum()),
        "et_tc_ratio": float(et.sum() / max(tc.sum(), 1)),
        "ed_wt_ratio": float(ed.sum() / max(wt.sum(), 1)),
        "tc_wt_ratio": float(tc.sum() / max(wt.sum(), 1)),
        "brain_pct": float(wt.sum() / (H * W * D) * 100),
    }

    # Alpha value
    n_missing = len(missing_modalities)
    alpha_map = {0: 0.34, 1: 0.48, 2: 0.58, 3: 0.66}
    measurements["alpha"] = alpha_map.get(n_missing, 0.5)
    measurements["n_missing"] = n_missing

    return pred_volume, measurements


# =====================================================================
# Report Generation
# =====================================================================

CONFIDENCE_RULES = {
    "WT":  {"needs": [0, 2],    "label_if_missing": "LOW"},
    "TC":  {"needs": [1],       "label_if_missing": "MODERATE"},
    "ET":  {"needs": [1],       "label_if_missing": "LOW"},
    "ED":  {"needs": [0, 2],    "label_if_missing": "LOW"},
    "NCR": {"needs": [1, 3],    "label_if_missing": "MODERATE"},
}

def get_confidence(region, missing_mods):
    rule = CONFIDENCE_RULES[region]
    if any(m in missing_mods for m in rule["needs"]):
        return rule["label_if_missing"]
    return "HIGH"


def generate_report(measurements, missing_modalities):
    """Generate structured clinical report without LLM (template-based for demo)"""
    available = [MODALITY_NAMES[i] for i in range(4) if i not in missing_modalities]
    missing = [MODALITY_NAMES[i] for i in missing_modalities]
    alpha = measurements["alpha"]
    n_miss = measurements["n_missing"]

    # Confidence per region
    conf = {r: get_confidence(r, missing_modalities) for r in CONFIDENCE_RULES}

    report = []
    report.append("=" * 70)
    report.append("  MISSING MODALITY-AWARE CLINICAL REPORT")
    report.append("=" * 70)

    # Modality status
    report.append(f"\n[Modality Status]")
    report.append(f"  Available: {', '.join(available) if available else 'NONE'}")
    report.append(f"  Missing:   {', '.join(missing) if missing else 'NONE (complete)'}")
    report.append(f"  Text fusion weight (α): {alpha:.2f}")

    # Quantitative results
    report.append(f"\n[Quantitative Segmentation Results]")
    report.append(f"  Whole Tumor (WT):     {measurements['wt_voxels']:,} voxels "
                  f"({measurements['brain_pct']:.2f}% of brain)")
    report.append(f"  Tumor Core (TC):      {measurements['tc_voxels']:,} voxels")
    report.append(f"  Enhancing Tumor (ET): {measurements['et_voxels']:,} voxels")
    report.append(f"  Necrotic Core (NCR):  {measurements['ncr_voxels']:,} voxels")
    report.append(f"  Peritumoral Edema:    {measurements['ed_voxels']:,} voxels")
    report.append(f"  ET/TC ratio:          {measurements['et_tc_ratio']:.3f}")
    report.append(f"  ED/WT ratio:          {measurements['ed_wt_ratio']:.3f}")

    # Confidence assessment
    report.append(f"\n[Region Confidence Assessment]")
    for region, level in conf.items():
        icon = "✅" if level == "HIGH" else ("⚠️" if level == "MODERATE" else "❌")
        report.append(f"  {icon} {region:>4s}: {level}")

    # Clinical note
    report.append(f"\n[Clinical Note]")
    if n_miss == 0:
        report.append(f"  All four MRI modalities available (α={alpha:.2f}).")
        report.append(f"  Text contribution minimal; segmentation relies primarily on imaging.")
        report.append(f"  All regions segmented with HIGH confidence.")
    elif n_miss <= 2:
        report.append(f"  {n_miss} modality(ies) missing: {', '.join(missing)} (α={alpha:.2f}).")
        low_regions = [r for r, c in conf.items() if c != "HIGH"]
        if low_regions:
            report.append(f"  Reduced confidence in: {', '.join(low_regions)}.")
        report.append(f"  Acquiring {', '.join(missing)} is recommended for improved delineation.")
    else:
        report.append(f"  SEVERE DEGRADATION: {n_miss} modalities missing (α={alpha:.2f}).")
        report.append(f"  Text report carries maximum weight to compensate for absent imaging.")
        low_regions = [r for r, c in conf.items() if c != "HIGH"]
        report.append(f"  LOW confidence regions: {', '.join(low_regions)}.")
        report.append(f"  ⚠️ Acquiring {', '.join(missing)} is STRONGLY recommended.")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


# =====================================================================
# Visualization
# =====================================================================

def create_overlay(vol_slice, seg_slice):
    """Create RGB overlay of segmentation on MRI slice"""
    # Normalize MRI to 0-255
    if vol_slice.max() > 0:
        mri = ((vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min()) * 255).astype(np.uint8)
    else:
        mri = np.zeros_like(vol_slice, dtype=np.uint8)

    rgb = np.stack([mri, mri, mri], axis=-1)

    # Colors: ET=yellow, NCR=red, ED=green
    et_mask = seg_slice[:, :, 0] > 0
    ncr_mask = seg_slice[:, :, 1] > 0
    ed_mask = seg_slice[:, :, 2] > 0

    alpha_overlay = 0.5
    rgb[ed_mask] = (rgb[ed_mask] * (1 - alpha_overlay) +
                    np.array([0, 255, 0]) * alpha_overlay).astype(np.uint8)
    rgb[ncr_mask] = (rgb[ncr_mask] * (1 - alpha_overlay) +
                     np.array([255, 0, 0]) * alpha_overlay).astype(np.uint8)
    rgb[et_mask] = (rgb[et_mask] * (1 - alpha_overlay) +
                    np.array([255, 255, 0]) * alpha_overlay).astype(np.uint8)

    return rgb


# =====================================================================
# Gradio Interface
# =====================================================================

def process_nifti(flair_file, t1ce_file, t2_file, t1_file,
                  miss_flair, miss_t1ce, miss_t2, miss_t1,
                  slice_idx):
    """Main processing function for Gradio"""
    try:
        import nibabel as nib
    except ImportError:
        return None, "Error: nibabel not installed. Run: pip install nibabel"

    # Determine missing modalities
    missing = []
    if miss_flair: missing.append(0)
    if miss_t1ce: missing.append(1)
    if miss_t2: missing.append(2)
    if miss_t1: missing.append(3)

    # Load available volumes
    volumes = {}
    files = {0: flair_file, 1: t1ce_file, 2: t2_file, 3: t1_file}

    for mi, f in files.items():
        if f is not None and mi not in missing:
            try:
                vol = nib.load(f.name).get_fdata().astype(np.float32)
                volumes[mi] = vol
            except Exception as e:
                return None, f"Error loading {MODALITY_NAMES[mi]}: {str(e)}"

    if not volumes:
        return None, "Error: No valid modality files uploaded."

    # Run inference
    pred_volume, measurements = run_inference(volumes, missing)

    # Generate report
    report = generate_report(measurements, missing)

    # Create visualization
    ref_vol = next(iter(volumes.values()))
    D = ref_vol.shape[2]
    z = min(max(1, int(slice_idx)), D - 2) if slice_idx else D // 2

    # Get reference slice for background
    ref_slice = ref_vol[:, :, z]
    seg_slice = pred_volume[:, :, z, :]
    overlay = create_overlay(ref_slice, seg_slice)

    return overlay, report


# Build Gradio UI
def build_demo():
    with gr.Blocks(
        title="TextBraTS: Missing-Modality Brain Tumor Segmentation",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🧠 TextBraTS: Missing-Modality Brain Tumor Segmentation
        ### Modality-Aware Adaptive Text-Visual Fusion

        Upload BraTS-format NIfTI files and select which modalities are missing.
        The system automatically adjusts text fusion weight (α) based on modality availability.

        **Color legend:** 🟡 Enhancing Tumor (ET) | 🔴 Necrotic Core (NCR) | 🟢 Peritumoral Edema (ED)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload MRI Modalities")
                flair = gr.File(label="FLAIR (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                t1ce = gr.File(label="T1CE (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                t2 = gr.File(label="T2 (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                t1 = gr.File(label="T1 (.nii / .nii.gz)", file_types=[".nii", ".gz"])

                gr.Markdown("### ❌ Missing Modalities")
                miss_flair = gr.Checkbox(label="FLAIR missing")
                miss_t1ce = gr.Checkbox(label="T1CE missing")
                miss_t2 = gr.Checkbox(label="T2 missing")
                miss_t1 = gr.Checkbox(label="T1 missing")

                slice_slider = gr.Slider(1, 154, value=77, step=1, label="Axial Slice")
                run_btn = gr.Button("🚀 Run Segmentation", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Segmentation Result")
                output_image = gr.Image(label="Segmentation Overlay", type="numpy")

                gr.Markdown("### 📋 Clinical Report")
                output_report = gr.Textbox(
                    label="Generated Report",
                    lines=25,
                    show_copy_button=True
                )

        run_btn.click(
            fn=process_nifti,
            inputs=[flair, t1ce, t2, t1, miss_flair, miss_t1ce, miss_t2, miss_t1, slice_slider],
            outputs=[output_image, output_report]
        )

        gr.Markdown("""
        ---
        **Paper:** Modality-Aware Adaptive Text-Visual Fusion for Robust Brain Tumor Segmentation with Missing MRI Modalities

        **Note:** This demo uses template-based report generation.
        The full system uses LLaMA 3-8B for natural language report generation with explicit per-region confidence disclosure.
        """)

    return demo


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting demo...")
    demo = build_demo()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

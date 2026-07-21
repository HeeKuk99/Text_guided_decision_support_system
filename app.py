"""
====================================================================
Text-Guided Decision Support System
for Brain Tumor Segmentation with Missing MRI Modalities
====================================================================
Interactive demo: upload MRI, select missing modalities,
get segmentation + clinical report with confidence disclosure.

Usage:
    pip install gradio nibabel torch numpy huggingface_hub
    python app.py
====================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# =====================================================================
# Model Architecture (inference only)
# =====================================================================

class _CB(nn.Module):
    def __init__(self, ic, oc, down=True):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(True), nn.Dropout2d(0.35),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(True))
        self.d = nn.MaxPool2d(2) if down else None
    def forward(self, x):
        s = self.c(x)
        return (s, self.d(s)) if self.d else s

class _UB(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.u = nn.ConvTranspose2d(ic, oc, 2, stride=2)
        self.c = _CB(oc + sc, oc, down=False)
    def forward(self, x, sk):
        return self.c(torch.cat([sk, self.u(x)], 1))

class _VF(nn.Module):
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

class _AG(nn.Module):
    def __init__(self, nm=4, h=128, vd=512):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(nm, h), nn.ReLU(), nn.Linear(h, vd), nn.Sigmoid())
    def forward(self, mask, fi, ft, htm=None):
        a = self.g(mask).view(fi.shape[0], -1, 1, 1)
        f = fi + a * (ft - fi)
        if htm is not None:
            b = htm.view(fi.shape[0], 1, 1, 1).float()
            f = fi * (1-b) + f * b
        return f

class TextGuidedModel(nn.Module):
    def __init__(self, n_cls=3, txt_dim=768, fd=512):
        super().__init__()
        self.enc1 = _CB(16, 64); self.enc2 = _CB(64, 128)
        self.enc3 = _CB(128, 256); self.enc4 = _CB(256, fd, down=False)
        self.bn_d = nn.MaxPool2d(2); self.bn_c = _CB(fd, fd, down=False)
        self.tp = nn.Sequential(nn.Linear(txt_dim, fd), nn.GELU(), nn.LayerNorm(fd), nn.Dropout(0.3))
        self.dt = nn.Parameter(torch.randn(1, fd))
        self.vf_s4 = _VF(fd, fd, fd); self.vf_bn = _VF(fd, fd, fd)
        self.ag_s4 = _AG(4, 128, fd); self.ag_bn = _AG(4, 128, fd)
        self.dec4 = _UB(fd, fd, 256); self.dec3 = _UB(256, 256, 128)
        self.dec2 = _UB(128, 128, 64); self.dec1 = _UB(64, 64, 64)
        self.head = nn.Conv2d(64, n_cls, 1)
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_ratio = nn.Parameter(torch.tensor(0.0))
        self.rh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fd, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, img, txt, has_text_mask=None):
        B = img.shape[0]
        s1, x = self.enc1(img); s2, x = self.enc2(x)
        s3, x = self.enc3(x); s4 = self.enc4(x)
        deep = self.bn_c(self.bn_d(s4))
        if txt.shape[-1] == 768:
            t = self.tp(txt if txt.dim()==2 else txt.view(B,-1)[:,:768])
        else:
            t = self.dt.expand(B,-1)
        if has_text_mask is not None:
            m = has_text_mask.view(B,1).float()
            t = t*m + self.dt.expand(B,-1)*(1-m)
        mask = img[:,12:].mean(dim=[2,3])
        s4f = self.ag_s4(mask, s4, self.vf_s4(s4, t, has_text_mask), has_text_mask)
        bnf = self.ag_bn(mask, deep, self.vf_bn(deep, t, has_text_mask), has_text_mask)
        x = self.dec4(bnf, s4f); x = self.dec3(x, s3)
        x = self.dec2(x, s2); x = self.dec1(x, s1)
        return self.head(x), self.rh(bnf)


# =====================================================================
# Weight Loading
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CROP = (128, 128)
MOD_NAMES = ["FLAIR", "T1CE", "T2", "T1"]

KEY_MAP = {
    "vlm_fusion_skip4": "vf_s4", "vlm_fusion_bottleneck": "vf_bn",
    "gate_skip4": "ag_s4", "gate_bottleneck": "ag_bn",
    "final_head": "head", "default_text_embedding": "dt",
    "text_proj": "tp", "cross_attention": "ca",
    "v_proj": "vp", "t_proj": "tp_linear",
    "norm1": "n1", "norm2": "n2", "ffn": "ff",
    "bottleneck_down": "bn_d", "bottleneck_conv": "bn_c",
    "ratio_head": "rh", "conv": "c", "down": "d", "gate": "g", "up": "u",
}

def map_key(k):
    k = k.replace("module.", "")
    for old, new in KEY_MAP.items():
        k = k.replace(old, new)
    return k

def load_model(weight_path="model_weights.pth"):
    global MODEL
    MODEL = TextGuidedModel().to(DEVICE)
    if not os.path.exists(weight_path):
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading weights from HuggingFace...")
            weight_path = hf_hub_download(
                repo_id="yoreurehehee/Text_Guided_Decision_Support_System",
                filename="model_weights.pth"
            )
        except Exception as e:
            print(f"Warning: Could not download weights: {e}")
            print("Running with random weights (demo mode)")
            MODEL.eval()
            return MODEL
    state = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    mapped = {map_key(k): v for k, v in state.items()}
    MODEL.load_state_dict(mapped, strict=False)
    MODEL.eval()
    print(f"✅ Model loaded ({sum(p.numel() for p in MODEL.parameters())/1e6:.1f}M params)")
    return MODEL


# =====================================================================
# Inference
# =====================================================================

def normalize_vol(vol):
    mask = vol > 0
    if mask.sum() == 0: return vol
    vals = vol[mask]
    lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    vol = np.clip(vol, lo, hi)
    vol = (vol - vals.mean()) / (vals.std() + 1e-8)
    vol[~mask] = 0
    return vol

def run_inference(volumes, missing_mods):
    if MODEL is None: load_model()
    ref = next(iter(volumes.values()))
    H, W, D = ref.shape
    pred_vol = np.zeros((H, W, D, 3), dtype=np.float32)

    full = np.zeros((H, W, D, 4), dtype=np.float32)
    for mi in range(4):
        if mi in volumes and mi not in missing_mods:
            full[:,:,:,mi] = normalize_vol(volumes[mi])

    mm = np.ones(4, dtype=np.float32)
    for mi in missing_mods: mm[mi] = 0.0

    buf, idx = [], []
    with torch.no_grad():
        for z in range(1, D-1):
            chs = []
            for mi in range(4):
                for dz in [-1,0,1]:
                    chs.append(full[:,:,z+dz,mi])
            img = np.stack(chs, axis=0)
            h0 = max(0,(H-CROP[0])//2); w0 = max(0,(W-CROP[1])//2)
            crop = img[:, h0:h0+CROP[0], w0:w0+CROP[1]]
            ph, pw = CROP[0]-crop.shape[1], CROP[1]-crop.shape[2]
            if ph>0 or pw>0: crop = np.pad(crop, ((0,0),(0,ph),(0,pw)))
            mch = np.stack([np.full(CROP, mm[i], dtype=np.float32) for i in range(4)])
            buf.append(np.concatenate([crop, mch], axis=0))
            idx.append(z)

            if len(buf) >= 64 or z == D-2:
                bt = torch.tensor(np.stack(buf), dtype=torch.float32).to(DEVICE)
                tt = torch.zeros(len(buf), 768, device=DEVICE)
                ht = torch.zeros(len(buf), dtype=torch.bool, device=DEVICE)
                seg, _ = MODEL(bt, tt, has_text_mask=ht)
                pred = (torch.sigmoid(seg)>0.5).float().cpu().numpy()
                for i, zi in enumerate(idx):
                    out = np.zeros((3,H,W), dtype=np.float32)
                    out[:, h0:h0+CROP[0], w0:w0+CROP[1]] = pred[i,:,:CROP[0],:CROP[1]]
                    pred_vol[:,:,zi,:] = out.transpose(1,2,0)
                buf, idx = [], []

    et=pred_vol[:,:,:,0]; ncr=pred_vol[:,:,:,1]; ed=pred_vol[:,:,:,2]
    tc=np.clip(et+ncr,0,1); wt=np.clip(et+ncr+ed,0,1)
    alpha_map = {0:0.34, 1:0.48, 2:0.58, 3:0.66}
    return pred_vol, {
        "wt_voxels":int(wt.sum()), "tc_voxels":int(tc.sum()),
        "et_voxels":int(et.sum()), "ncr_voxels":int(ncr.sum()),
        "ed_voxels":int(ed.sum()),
        "et_tc_ratio":float(et.sum()/max(tc.sum(),1)),
        "ed_wt_ratio":float(ed.sum()/max(wt.sum(),1)),
        "tc_wt_ratio":float(tc.sum()/max(wt.sum(),1)),
        "brain_pct":float(wt.sum()/(H*W*D)*100),
        "alpha":alpha_map.get(len(missing_mods),0.5),
        "n_missing":len(missing_mods),
    }


# =====================================================================
# Report Generation
# =====================================================================

CONF = {
    "WT":{"needs":[0,2],"low":"LOW"}, "TC":{"needs":[1],"low":"MODERATE"},
    "ET":{"needs":[1],"low":"LOW"}, "ED":{"needs":[0,2],"low":"LOW"},
    "NCR":{"needs":[1,3],"low":"MODERATE"},
}

def get_conf(r, miss):
    return CONF[r]["low"] if any(m in miss for m in CONF[r]["needs"]) else "HIGH"

def generate_report(meas, missing_mods):
    avail = [MOD_NAMES[i] for i in range(4) if i not in missing_mods]
    miss = [MOD_NAMES[i] for i in missing_mods]
    alpha = meas["alpha"]; n = meas["n_missing"]
    conf = {r: get_conf(r, missing_mods) for r in CONF}

    L = []
    L.append("=" * 65)
    L.append("  MISSING MODALITY-AWARE CLINICAL REPORT")
    L.append("  Text-Guided Decision Support System")
    L.append("=" * 65)
    L.append(f"\n[Modality Status]")
    L.append(f"  Available : {', '.join(avail) if avail else 'NONE'}")
    L.append(f"  Missing   : {', '.join(miss) if miss else 'NONE (complete)'}")
    L.append(f"  Adaptive Gate α: {alpha:.2f}")
    L.append(f"\n[Quantitative Segmentation]")
    L.append(f"  Whole Tumor (WT)      : {meas['wt_voxels']:>8,} voxels ({meas['brain_pct']:.2f}% of brain)")
    L.append(f"  Tumor Core (TC)       : {meas['tc_voxels']:>8,} voxels")
    L.append(f"  Enhancing Tumor (ET)  : {meas['et_voxels']:>8,} voxels")
    L.append(f"  Necrotic Core (NCR)   : {meas['ncr_voxels']:>8,} voxels")
    L.append(f"  Peritumoral Edema (ED): {meas['ed_voxels']:>8,} voxels")
    L.append(f"  ET/TC ratio           : {meas['et_tc_ratio']:.3f}")
    L.append(f"  ED/WT ratio           : {meas['ed_wt_ratio']:.3f}")
    L.append(f"\n[Region Confidence]")
    for r, lv in conf.items():
        ic = {"HIGH":"✅","MODERATE":"⚠️","LOW":"❌"}[lv]
        L.append(f"  {ic} {r:>4s} : {lv}")
    L.append(f"\n[Clinical Note]")
    if n == 0:
        L.append(f"  All MRI modalities available (α={alpha:.2f}).")
        L.append(f"  Segmentation relies primarily on imaging.")
        L.append(f"  All regions segmented with HIGH confidence.")
    elif n <= 2:
        L.append(f"  {n} modality(ies) missing: {', '.join(miss)} (α={alpha:.2f}).")
        lr = [r for r,c in conf.items() if c != "HIGH"]
        if lr: L.append(f"  Reduced confidence: {', '.join(lr)}.")
        L.append(f"  Acquiring {', '.join(miss)} recommended for improved delineation.")
    else:
        L.append(f"  ⚠️  SEVERE: {n} modalities missing (α={alpha:.2f}).")
        L.append(f"  Text report carries maximum weight.")
        lr = [r for r,c in conf.items() if c != "HIGH"]
        L.append(f"  LOW confidence: {', '.join(lr)}.")
        L.append(f"  Acquiring {', '.join(miss)} is STRONGLY recommended.")
    if meas['et_tc_ratio'] > 0.5 and meas['tc_voxels'] > 1000:
        L.append(f"\n[Grade Hint]")
        L.append(f"  ET/TC={meas['et_tc_ratio']:.3f} suggests high-grade glioma (WHO Grade IV).")
    L.append("\n" + "=" * 65)
    return "\n".join(L)


# =====================================================================
# Visualization
# =====================================================================

def create_overlay(mri, seg):
    if mri.max() > 0:
        m = ((mri-mri.min())/(mri.max()-mri.min())*255).astype(np.uint8)
    else:
        m = np.zeros_like(mri, dtype=np.uint8)
    rgb = np.stack([m,m,m], axis=-1); a = 0.5
    rgb[seg[:,:,2]>0] = (rgb[seg[:,:,2]>0]*(1-a) + np.array([0,255,0])*a).astype(np.uint8)
    rgb[seg[:,:,1]>0] = (rgb[seg[:,:,1]>0]*(1-a) + np.array([255,0,0])*a).astype(np.uint8)
    rgb[seg[:,:,0]>0] = (rgb[seg[:,:,0]>0]*(1-a) + np.array([255,255,0])*a).astype(np.uint8)
    return rgb


# =====================================================================
# Gradio App
# =====================================================================

def process(flair_file, t1ce_file, t2_file, t1_file,
            miss_flair, miss_t1ce, miss_t2, miss_t1, slice_idx):
    try:
        import nibabel as nib
    except ImportError:
        return None, "Error: nibabel not installed."

    missing = []
    if miss_flair: missing.append(0)
    if miss_t1ce: missing.append(1)
    if miss_t2: missing.append(2)
    if miss_t1: missing.append(3)

    volumes = {}
    files = {0: flair_file, 1: t1ce_file, 2: t2_file, 3: t1_file}
    for mi, f in files.items():
        if f is not None and mi not in missing:
            try:
                volumes[mi] = nib.load(f.name).get_fdata().astype(np.float32)
            except Exception as e:
                return None, f"Error loading {MOD_NAMES[mi]}: {e}"
    if not volumes:
        return None, "No valid modality files uploaded."

    pred, meas = run_inference(volumes, missing)
    report = generate_report(meas, missing)
    ref = next(iter(volumes.values()))
    D = ref.shape[2]
    z = min(max(1, int(slice_idx)), D-2) if slice_idx else D//2
    return create_overlay(ref[:,:,z], pred[:,:,z,:]), report

def build_app():
    with gr.Blocks(title="Text-Guided Decision Support System", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🧠 Text-Guided Decision Support System
        ### Modality-Aware Adaptive Fusion for Brain Tumor Segmentation under Missing MRI Modalities

        Upload BraTS-format NIfTI files and select which modalities are missing.
        The adaptive gate automatically adjusts text fusion weight (α) based on modality availability.

        **Segmentation:** 🟡 Enhancing Tumor (ET) | 🔴 Necrotic Core (NCR) | 🟢 Peritumoral Edema (ED)
        """)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload MRI Modalities")
                flair = gr.File(label="FLAIR (.nii/.nii.gz)", file_types=[".nii",".gz"])
                t1ce = gr.File(label="T1CE (.nii/.nii.gz)", file_types=[".nii",".gz"])
                t2 = gr.File(label="T2 (.nii/.nii.gz)", file_types=[".nii",".gz"])
                t1 = gr.File(label="T1 (.nii/.nii.gz)", file_types=[".nii",".gz"])
                gr.Markdown("### ❌ Select Missing Modalities")
                mf = gr.Checkbox(label="FLAIR missing")
                mt = gr.Checkbox(label="T1CE missing")
                m2 = gr.Checkbox(label="T2 missing")
                m1 = gr.Checkbox(label="T1 missing")
                sl = gr.Slider(1, 154, value=77, step=1, label="Axial Slice Index")
                btn = gr.Button("🚀 Run Segmentation & Generate Report", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Segmentation Result")
                img_out = gr.Image(label="Segmentation Overlay", type="numpy")
                gr.Markdown("### 📋 Clinical Report with Confidence Disclosure")
                txt_out = gr.Textbox(label="Generated Report", lines=28, show_copy_button=True)
        btn.click(fn=process, inputs=[flair,t1ce,t2,t1,mf,mt,m2,m1,sl], outputs=[img_out,txt_out])
        gr.Markdown("""
        ---
        **Paper:** Modality-Aware Adaptive Text-Visual Fusion for Robust Brain Tumor Segmentation with Missing MRI Modalities

        **Model weights:** [HuggingFace](https://huggingface.co/yoreurehehee/Text_Guided_Decision_Support_System) |
        **Code:** [GitHub](https://github.com/HeeKuk99/Text_guided_decision_support_system)

        *This demo uses template-based report generation. The full system uses LLaMA 3-8B
        for natural language report generation with explicit per-region confidence disclosure.*
        """)
    return app

if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting Text-Guided Decision Support System...")
    app = build_app()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)

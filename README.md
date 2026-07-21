

# TextBraTS: Modality-Aware Adaptive Text-Visual Fusion

Interactive demo for missing-modality brain tumor segmentation with adaptive text-visual fusion.

## Usage
1. Upload BraTS-format NIfTI files (FLAIR, T1CE, T2, T1)
2. Check which modalities are missing
3. Click "Run Segmentation"
4. View segmentation overlay and generated clinical report

## Features
- **Adaptive Gate**: Automatically adjusts text fusion weight (α) based on modality availability
- **15 scenarios**: Supports any combination of missing modalities
- **Clinical report**: Per-region confidence disclosure (HIGH/MODERATE/LOW)
- **Color legend**: 🟡 ET | 🔴 NCR | 🟢 ED

## Citation
```bibtex
@article{textbrats2026,
  title={Modality-Aware Adaptive Text-Visual Fusion for Robust Brain Tumor Segmentation with Missing MRI Modalities},
  year={2026}
}
```

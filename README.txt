# Lightweight Multi-Frame Integration for YOLOv7 (ECMR 2025)

This repo turns single-frame YOLOv7/YOLOv7-tiny into a **multi-frame** detector by stacking consecutive frames along the channel axis and supervising **only the latest frame** (weak supervision). Two first-layer variants are supported: **Early Fusion** (standard conv over all stacked channels) and **Late Fusion** (*Grouped Convolution*: one group per frame, merged later). The approach improves robustness to blur/occlusion with minimal parameter growth and real-time speed. See the paper for details.

---

## 1) What’s in this repo?

- **Multi-frame I/O path** for training/validation/inference: frames are stacked as **3×n channels** and passed through the model; only the most recent frame in each stack is labeled/supervised.
- **Minimal model edits**: only the first (and immediately following) conv layers are adapted for extra input channels; the neck/head remain standard YOLOv7.
- **Fusion strategies**:
  - **Early Fusion (EF-Multi)** — *recommended*: first conv sees `3n` channels and learns across frames jointly.
  - **Late Fusion (GC-Multi)** — optional ablation: **grouped conv** with *n* groups in the first conv (one group per frame), features concatenated and fused later.
- **Dataloaders**: stack neighboring or **stepped** frames; optional `PreStackedLoadImagesAndLabels` to precompute stacks; augmentations are synchronized across the stack to preserve alignment.
- **Inference**: `detect.py` buffers recent frames and auto-infers *n* from `in_channels/3`.

> **Note:** This repository is built upon the [YOLOv7](https://github.com/WongKinYiu/yolov7) implementation by Wang et al., with modifications for lightweight multi-frame input and weakly supervised training.


---

## 2) Method overview

**Goal.** Use *n* consecutive frames {\(I_{t-n+1}, …, I_t\)} as input but predict detections **only for \(I_t\)**. This injects temporal cues at the pixel level without heavy temporal modules (RNNs/flow/attention).

### 2.1 Early Fusion (EF-Multi)
- Change the first conv to accept `3n` input channels.
- **Weight init**: tile the 1-frame weights *n* times along the input-channel dim (alternatively scale by `1/n` to preserve the initial response).
- Subsequent layers remain unchanged.

### 2.2 Late Fusion / Grouped Convolution (GC-Multi)
- Set `groups = n` in the first conv so each group processes **one frame** (3 channels).
- Initialize each group with a copy of the single-frame weights.
- The **next conv** is adjusted to consume the concatenated feature maps (expanded channel dimension).
- Typically lower accuracy than Early Fusion but useful to isolate the effect of cross-frame mixing in the first layer.


### 2.3 Training (weak supervision)
- Only the labels of the  **latest frame** in each stack  contributes to the YOLO loss; earlier frames are unlabeled context (meaning their labels are ignored in the implementation).
- Augmentations (resize/crop/flip/warp) must be **identical across all frames** in a stack.

- Standard YOLO losses and schedulers apply.

---

## 3) Practical recommendations (paper summary)

- **How many frames?** Start with **n = 3**. For static cameras/crowded scenes, try **5 or 7**. Too many (e.g., 9) can hurt due to noise.
- **How many frames?** Start with **n = 3**. For static cameras/crowded scenes, try **5 or 7**. The optimal choice **highly depends on the frame rate and how quickly the scene changes**. Too many frames can hurt due to noise.
- **Temporal step sampling** can match or beat dense stacking: e.g., **3 frames with step=3** (indices like `[-6, -3, 0]`) or **4 frames with step=2** (indices like `[-6, -4, -2, 0]`) or **5 frames with step=2** can perform on par with/above 7 adjacent frames while using less computational effort.
- **Fusion choice**: **Early Fusion**  outperforms **Grouped Convolution** across settings in our ablations.
- **Model scale**: Gains are largest for **YOLOv7-tiny**; multi-frame tiny closes much of the gap to single-frame YOLOv7.
- **Efficiency**: Parameters/GFLOPs increase only marginally vs 1-frame; throughput remains real-time on embedded GPUs.

---

## 4) Quick start

### Environment
- Standard YOLOv7 dependencies (PyTorch + CUDA). Use the repo’s environment as you would for vanilla YOLOv7.

### Datasets to start with
- [**MOT20Det**](https://motchallenge.net/data/MOT20Det/): keep files in **temporal order** within each sequence; convert into yolo format; split train/val/test as needed.



### Choose the number of frames
- Default to **n=3**. Try **n=5/7** for static scenes; prefer **stepped sampling** when motion is fast.

---

## 5) Training

### A) Early Fusion (recommended)

**Idea.** Stack frames on the channel axis; first conv takes `3n` channels. Initialize by repeating 1-frame weights *n* times, scaled by `1/n`. Details implemetation in script  `utils/datasets.py`.

**Command (example, YOLOv7-tiny, n=3):**

```bash
  python train.py  \ 
  --weights yolov7-tiny-3frames.pt \ 
  --data data/your_dataset_meta.yaml  \ 
  --cfg cfg/training/yolov7-tiny_group3_for_1st_layer.yaml \
  --hyp data/hyp.scratch.custom.yaml  \
  --n-frames 3
```

Notes:
- The loader stacks **neighboring frames** by default; augmentations are minimal and synchronized across frames.
- Passing **1-frame weights** triggers first-conv adaptation as described; you can also start from a multi-frame checkpoint.
- Sample file see in `cfg/training/yolov7_expand_tile_2nd_layer.yaml`.

**Temporal step sampling.** To emulate stepped setups, index frames with gaps, e.g., `[-6, -3, 0]` for **3@step3** or `[-4, -2, 0, +2, +4]` for **5@step2** (when available). Need to be manually set in script  `utils/datasets.py` with parameter `step` in method `def __getitem__(self, index)` under `class LoadImagesAndLabels(Dataset)`.  


### B) Late Fusion / Grouped Convolution 

**Idea.** First conv uses `groups = n` so **each group sees one frame** (3 channels). Concatenate group features after the first layer and fuse with the next conv.

**How to enable.**
- In your model YAML / first Conv block: `in_channels = 3*n`, `groups = n`.
- Keep per-group `out_channels` equal to the single-frame setting; the concatenation yields the same total `out_channels` as Early Fusion.
- Initialize each group with a copy of the 1-frame kernel weights. Adjust the **next** conv to accept the expanded channels.
- Sample file see in `cfg/training/yolov7-tiny_group3_for_1st_layer.yaml`.

> Expect **lower accuracy** than Early Fusion; include for completeness in ablations.

---

## 6) Validation / Testing

```bash
python test.py --data <your.yaml> --weights <ckpt>.pt --img-size 640 \
  --batch-size 8 --n-frames 3 --task val

python test.py \
 --data data/your_dataset_meta.yaml \ 
 --conf 0.001 \ --iou 0.65  \
 --weights yolov7-tiny-3frames.pt \ 
 --n-frame 3  
```

- Validation stacks frames and **reports metrics on the latest frame** in each stack. Empty-target batches are skipped.

---


## 8) Tips, efficiency & troubleshooting

- **Memory/throughput** scale roughly linearly with *n* (channels). Reduce batch or image size if needed.
- **Params/GFLOPs** increase only marginally from 1→multi-frame; embedded GPUs remain real-time for tiny.
- **Debug**: set `DEBUGGING=true` to dump per-frame PNGs; optional profiling prints stacking/augment timings for `PreStackedLoadImagesAndLabels`.

---

## 9) Reproducing common ablations

- **#Frames**: n ∈ {1, 3, 5, 7, 9}; expect peaks around 7 on static/crowded scenes and ~3 on fast-moving ones.
- **Step sampling**: try 3@step3, 5@step2, or mixed indices like `[-6, -3, -1, 0]`.
- **Fusion**: compare **EF-Multi** vs **GC-Multi** at the same n/step.
- **Scale**: repeat on YOLOv7-tiny and YOLOv7.

---

## 10) Citation

If you use this repo, please cite:

> [**Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos**](https://arxiv.org/pdf/2506.20550).  
> Quan, Kiefer, Messmer, Zell — ECMR 2025.

---

## 11) Acknowledgments
This work is built upon the official [YOLOv7](https://github.com/WongKinYiu/yolov7) repository by Wang et al.
We thank the authors for making their code publicly available.
The original YOLOv7 codebase is distributed under the GPL-3.0 license, which this repository follows.





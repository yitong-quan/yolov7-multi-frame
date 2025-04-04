# please run with '$ python viz-filter-feature-map_chatgpt_version.py --model yolov7.pt --image inference/images/image2.jpg --out run/output_maps'

import torch
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import math


# Patch nn.Upsample to remove recompute_scale_factor (for older PyTorch versions)
class PatchedUpsample(nn.Upsample):
    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)


def patch_upsample_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Upsample):
            setattr(module, name, PatchedUpsample(size=child.size, scale_factor=child.scale_factor, mode=child.mode,
                                                  align_corners=child.align_corners))
        else:
            patch_upsample_layers(child)


def patch_upsample_forward_only(module):
    for child in module.children():
        if isinstance(child, nn.Upsample):
            def safe_forward(self, input):
                return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

            child.forward = safe_forward.__get__(child, nn.Upsample)
        else:
            patch_upsample_forward_only(child)


# Extract feature maps
def register_hooks(model, feature_maps):
    def hook_fn(module, input, output):
        feature_maps.append(output)

    for layer in model.model:
        # if isinstance(layer, (nn.Conv2d, nn.Sequential, nn.Upsample)):
        #     layer.register_forward_hook(hook_fn)
        layer.register_forward_hook(hook_fn)


# def visualize_feature_maps(feature_maps, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for i, fmap in enumerate(feature_maps):
#         fmap = fmap[0]  # batch index 0
#         num_channels = fmap.shape[0]
#
#         grid_size = int(np.ceil(np.sqrt(num_channels)))
#         plt.figure(figsize=(grid_size, grid_size))
#         for j in range(num_channels):
#             plt.subplot(grid_size, grid_size, j + 1)
#             plt.imshow(fmap[j].detach().cpu().numpy(), cmap='gray')
#             plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"feature_map_{i}.png"))
#         plt.close()


def visualize_feature_maps_slow(feature_maps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # batch index 0 → shape [C, H, W]
        num_channels = fmap.shape[0]
        h, w = fmap.shape[1], fmap.shape[2]

        grid_cols = int(np.ceil(np.sqrt(num_channels)))
        grid_rows = int(np.ceil(num_channels / grid_cols))

        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols, grid_rows))
        axs = axs.flatten()

        for j in range(num_channels):
            channel_img = fmap[j].detach().cpu().numpy()
            axs[j].imshow(channel_img, cmap='gray')
            axs[j].axis('off')

        for j in range(num_channels, len(axs)):
            axs[j].axis('off')

        plt.tight_layout(pad=0.1)
        out_path = os.path.join(output_dir, f"layer_{i}_shape_{h}x{w}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()


def visualize_feature_maps_dark(feature_maps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # shape [C, H, W]
        normed = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # normalize to [0, 1]

        grid_img = make_grid(normed.unsqueeze(1), nrow=math.ceil(math.sqrt(fmap.shape[0])), padding=1, normalize=False)
        grid_img_pil = TF.to_pil_image(grid_img.squeeze(0).cpu())

        out_path = os.path.join(output_dir, f"layer_{i}_shape_{fmap.shape[1]}x{fmap.shape[2]}.png")
        grid_img_pil.save(out_path)


def visualize_feature_maps(feature_maps, output_dir, max_channels=640000):
    os.makedirs(output_dir, exist_ok=True)

    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # shape [C, H, W]

        # Limit number of channels for visualization
        fmap = fmap[:max_channels]

        # Normalize each channel independently to [0,1]
        normed_channels = []
        for c in fmap:
            c = c.detach().cpu()
            c_min, c_max = c.min(), c.max()
            if (c_max - c_min) > 1e-5:
                c_norm = (c - c_min) / (c_max - c_min)
            else:
                c_norm = torch.zeros_like(c)
            normed_channels.append(c_norm)

        normed = torch.stack(normed_channels)  # [C, H, W]

        # Make a grid of images with white padding
        grid_img = make_grid(normed.unsqueeze(1), nrow=int(math.sqrt(normed.shape[0])), padding=1, pad_value=1.0)

        # Convert to PIL and save
        grid_img_pil = TF.to_pil_image(grid_img.squeeze(0))
        out_path = os.path.join(output_dir, f"layer_{i}_shape_{fmap.shape[1]}x{fmap.shape[2]}.png")
        grid_img_pil.save(out_path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = torch.load(args.model, map_location=device)
    model = model['model'].to(device).eval()
    patch_upsample_forward_only(model)
    # patch_upsample_layers(model)

    # Load and preprocess image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((640, 640)),  # change if needed
        transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device).half()
    # img_tensor_repeat = img_tensor.repeat(1, 3, 1, 1)

    # Register hooks
    feature_maps = []
    register_hooks(model, feature_maps)

    # Forward pass
    with torch.no_grad():
        model(img_tensor)

    # Visualize
    visualize_feature_maps(feature_maps, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to YOLOv7 .pt model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--out', default='feature_maps', help='Output folder')
    args = parser.parse_args()
    main(args)

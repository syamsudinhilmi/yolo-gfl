from typing import List
import math
import torch
import torch.nn as nn

from block import C3Ghost, C3k2, GhostConv
from conv import Concat, Conv
from head import Detect


class YOLOGFL(nn.Module):
    """YOLO-GFL: GhostNet Fire Lightweight Architecture for Detection."""

    def __init__(self, nc: int = 2, scale: str = 'n'):
        """Initialize YOLO-GFL model with number of classes and scale."""
        super().__init__()
        self.nc = nc

        # Scale configurations [depth, width, max_channels] - Updated to match Ultralytics
        scales = {
            'n': [0.50, 0.25, 1024],  # Matching the YAML config
            's': [0.50, 0.50, 1024],
            'm': [0.67, 0.75, 1024],
            'l': [1.0, 1.0, 1024],
            'x': [1.33, 1.25, 1024]
        }

        depth_multiple, width_multiple, max_channels = scales[scale]

        # Channel calculations with width multiplier - exact matching
        base_channels = [64, 128, 256, 512, 1024]
        ch = [max(round(c * width_multiple), 1) if c * width_multiple >= 1 else 1 for c in base_channels]
        ch = [min(c, max_channels) for c in ch]

        # For 'n' scale: [16, 32, 64, 128, 256]
        print(f"Channels: {ch}")

        # Repeat calculations with depth multiplier
        def make_divisible(x):
            return max(round(x * depth_multiple), 1)

        # Build model according to YAML configuration
        # Backbone layers
        self.backbone = nn.ModuleList([
            Conv(3, ch[0], 3, 2),  # 0-P1/2
            Conv(ch[0], ch[1], 3, 2),  # 1-P2/4
            C3k2(ch[1], ch[1], make_divisible(2)),  # 2
            Conv(ch[1], ch[2], 3, 2),  # 3-P3/8
            C3k2(ch[2], ch[2], make_divisible(3)),  # 4
            Conv(ch[2], ch[3], 3, 2),  # 5-P4/16
            C3Ghost(ch[3], ch[3], make_divisible(4)),  # 6
            Conv(ch[3], ch[4], 3, 2),  # 7-P5/32
            C3Ghost(ch[4], ch[4], make_divisible(2)),  # 8
        ])

        # Head layers following exact YAML configuration
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 9
            Concat(1),  # 10 - cat backbone P4
            C3Ghost(ch[3] + ch[4], ch[3], make_divisible(2)),  # 11
            nn.Upsample(scale_factor=2, mode='nearest'),  # 12
            Concat(1),  # 13 - cat backbone P3
            C3Ghost(ch[2] + ch[3], ch[2], make_divisible(2)),  # 14
            GhostConv(ch[2], ch[2], 3, 2),  # 15
            Concat(1),  # 16 - cat head P4
            C3Ghost(ch[2] + ch[3], ch[3], make_divisible(2)),  # 17
            GhostConv(ch[3], ch[3], 3, 2),  # 18
            Concat(1),  # 19 - cat head P5
            C3k2(ch[3] + ch[4], ch[4], make_divisible(2)),  # 20
            Detect(nc, (ch[2], ch[3], ch[4]))  # 21 - Detect(P3, P4, P5)
        ])

        # Store channels for reference
        self.ch = ch
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through YOLO-GFL model."""
        # Store features for skip connections
        features = {}

        # Backbone forward pass
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 8]:  # Store P3, P4, P5 features (after layers 4, 6, 8)
                features[i] = x

        # Head forward pass following YAML structure
        # 9: Upsample
        x = self.head[0](x)
        # 10: Concat with P4 (from layer 6)
        x = self.head[1]([x, features[6]])
        # 11: C3Ghost
        x = self.head[2](x)
        p4_out = x

        # 12: Upsample
        x = self.head[3](x)
        # 13: Concat with P3 (from layer 4)
        x = self.head[4]([x, features[4]])
        # 14: C3Ghost
        x = self.head[5](x)
        p3_out = x

        # 15: GhostConv
        x = self.head[6](x)
        # 16: Concat with P4
        x = self.head[7]([x, p4_out])
        # 17: C3Ghost
        x = self.head[8](x)
        p4_final = x

        # 18: GhostConv
        x = self.head[9](x)
        # 19: Concat with P5 (from layer 8)
        x = self.head[10]([x, features[8]])
        # 20: C3k2
        x = self.head[11](x)
        p5_final = x

        # 21: Detect
        detections = self.head[12]([p3_out, p4_final, p5_final])

        return detections

    def model_info(self, verbose=True, imgsz=640):
        """Print detailed model summary including nested structure."""

        def count_layers(module, layer_count=[0]):
            """Recursively count all layers."""
            for child in module.children():
                if len(list(child.children())) > 0:
                    count_layers(child, layer_count)
                else:
                    layer_count[0] += 1
            return layer_count[0]

        def get_layer_info(module, name="", depth=0):
            """Get detailed layer information."""
            info = []
            indent = "  " * depth

            if len(list(module.children())) == 0:  # Leaf node
                param_count = sum(p.numel() for p in module.parameters())
                info.append(f"{indent}{name}: {module.__class__.__name__} - {param_count:,} params")
            else:
                info.append(f"{indent}{name}: {module.__class__.__name__}")
                for i, (child_name, child_module) in enumerate(module.named_children()):
                    info.extend(get_layer_info(child_module, f"{child_name}", depth + 1))

            return info

        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_layers = count_layers(self)

        # Calculate GFLOPs
        gflops = self._calculate_gflops(imgsz)

        # Print summary in Ultralytics format
        print(f"YOLO-GFL summary: {total_layers} layers, {total_params:,} parameters, {gflops:.1f} GFLOPs")

        if verbose:
            print(f"\nModel: {self.__class__.__name__}")
            print(f"Channels: {self.ch}")
            print(f"Depth multiple: {self.depth_multiple}")
            print(f"Width multiple: {self.width_multiple}")
            print(f"\nDetailed Architecture:")

            # Print backbone structure
            print("Backbone:")
            for i, layer in enumerate(self.backbone):
                layer_info = get_layer_info(layer, f"  [{i}]", 1)
                for line in layer_info:
                    print(line)

            # Print head structure
            print("\nHead:")
            for i, layer in enumerate(self.head):
                layer_info = get_layer_info(layer, f"  [{i + 9}]", 1)
                for line in layer_info:
                    print(line)

            print(
                f"\nTotal: {total_layers} layers, {total_params:,} parameters, {trainable_params:,} gradients, {gflops:.1f} GFLOPs")

    def _calculate_gflops(self, imgsz=640):
        """Calculate GFLOPs more accurately."""
        try:
            from thop import profile
            x = torch.randn(1, 3, imgsz, imgsz)
            flops, _ = profile(self, inputs=(x,), verbose=False)
            return flops / 1e9
        except ImportError:
            # Fallback estimation
            return self._estimate_gflops_simple(imgsz)

    def _estimate_gflops_simple(self, imgsz=640):
        """Simple GFLOPs estimation."""
        total_ops = 0

        # Estimate based on channel sizes and typical YOLO operations
        h, w = imgsz, imgsz

        # Backbone estimation
        for i, ch_out in enumerate(self.ch):
            if i == 0:
                ch_in = 3
            else:
                ch_in = self.ch[i - 1]

            # Rough estimation for conv operations at each scale
            scale = 2 ** (i + 1)
            fh, fw = h // scale, w // scale

            # Basic conv ops estimation
            ops = fh * fw * ch_in * ch_out * 9  # 3x3 conv
            total_ops += ops

            # Additional ops for C3/C3Ghost blocks
            if i > 1:  # Has C3 blocks
                total_ops += ops * 2  # Approximate C3 overhead

        # Head operations (approximately 30% of backbone)
        total_ops += total_ops * 0.3

        return total_ops / 1e9

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers."""
        print("Fusing layers...")
        fused_count = 0

        def fuse_conv_bn(conv, bn):
            """Fuse convolution and batchnorm."""
            fusedconv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
                bias=True
            ).requires_grad_(False).to(conv.weight.device)

            # Prepare filters
            w_conv = conv.weight.clone().view(conv.out_channels, -1)
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
            fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

            # Prepare spatial bias
            b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

            return fusedconv

        # Recursively fuse all Conv modules
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
                fused_count += 1

        print(f"Fused {fused_count} layers")
        return self


def create_yolo_gfl(nc: int = 2, scale: str = 'n') -> YOLOGFL:
    """Create YOLO-GFL model instance."""
    model = YOLOGFL(nc=nc, scale=scale)
    return model


def test_yolo_gfl():
    """Test YOLO-GFL model with detailed output."""
    print("\nCreating YOLO-GFL model...")
    model = create_yolo_gfl(nc=2, scale='n')

    # Test input
    x = torch.randn(1, 3, 640, 640)

    print("YOLO-GFL Model Architecture Analysis")

    # Detailed model info
    model.model_info(verbose=True)

    print("\nForward Pass Test")

    # Test forward pass
    with torch.no_grad():
        outputs = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Number of detection heads: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  Detection head {i + 1}: {out.shape}")

    print("\nModel Fusion Test")

    # Test fusion
    fused_model = model.fuse()
    with torch.no_grad():
        fused_outputs = fused_model(x)
        print("Fused model performance:")
        for i, out in enumerate(fused_outputs):
            print(f"  Fused detection head {i + 1}: {out.shape}")

    # Compare parameter counts
    print("\nComparison with Target (Ultralytics)")
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = sum(1 for _ in model.modules())

    print(f"Current Implementation:")
    print(f"  Layers: {total_layers}")
    print(f"  Parameters: {total_params:,}")
    print(f"\nTarget (Ultralytics):")
    print(f"  Layers: 198")
    print(f"  Parameters: 1,616,270")
    print(f"  GFLOPs: 4.7")


if __name__ == "__main__":
    test_yolo_gfl()
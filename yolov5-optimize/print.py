import torch
from copy import deepcopy

# ===============================
# Load trained model (best.pt)
# ===============================
ckpt = torch.load("best.pt", map_location="cpu")

model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
model = model.float().fuse().eval()   # IMPORTANT: match validation params

print("\nYOLOv5 model summary (from best.pt)\n")
print(f"{'from':>6} {'n':>3} {'params':>12}  {'module':<30} arguments")
print("-" * 80)


# ===============================
# helper: count params
# ===============================
def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ===============================
# helper: extract arguments
# ===============================
def get_args(m):

    name = m.__class__.__name__

    # Conv
    if hasattr(m, "conv"):
        c1 = m.conv.in_channels
        c2 = m.conv.out_channels
        k = m.conv.kernel_size
        s = m.conv.stride
        return f"[{c1}, {c2}, k={k}, s={s}]"

    # C3 / BottleneckCSP-like
    if hasattr(m, "cv1") and hasattr(m.cv1, "conv"):
        c1 = m.cv1.conv.in_channels
        c2 = m.cv2.conv.out_channels
        return f"[{c1}, {c2}]"

    # CoordAtt / Attention blocks
    if hasattr(m, "conv1"):
        try:
            c1 = m.conv1.in_channels
            c2 = m.conv1.out_channels
            return f"[{c1}, {c2}]"
        except:
            return "[]"

    # SPPF
    if name == "SPPF":
        return f"[k={m.k}]"

    # Upsample
    if "Upsample" in name:
        return f"[scale_factor={m.scale_factor}]"

    # Concat
    if name == "Concat":
        return "[]"

    # Detect
    if name == "Detect":
        return f"[nc={m.nc}]"

    return "[]"


# ===============================
# Print layer table
# ===============================
total_params = 0

for i, m in enumerate(model.model):

    params = count_params(m)
    total_params += params

    # YOLOv5 stores source connection here
    f = getattr(m, "f", -1)
    n = getattr(m, "n", 1)

    module_name = m.__class__.__name__
    args = get_args(m)

    print(f"{str(f):>6} {n:>3} {params:12,}  {module_name:<30} {args}")

print("-" * 80)
print(f"Total params: {total_params:,}")

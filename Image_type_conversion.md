# 🖼️ Image Format Conversion in Python

## 📌 Common Image Types

| Type           | Library        | Shape     | Channel Order | Value Range                    |
| -------------- | -------------- | --------- | ------------- | ------------------------------ |
| `PIL.Image`    | Pillow         | (H, W, C) | RGB           | 0–255 (uint8)                  |
| `np.ndarray`   | NumPy          | (H, W, C) | RGB or BGR    | 0–255 (uint8)                  |
| `cv2 image`    | OpenCV (NumPy) | (H, W, C) | BGR           | 0–255 (uint8)                  |
| `torch.Tensor` | PyTorch        | (C, H, W) | RGB           | 0–1 (float32) or 0–255 (uint8) |

---

## 🔁 Conversions

### 1. **PIL.Image ↔ NumPy (RGB)**

#### a. PIL → NumPy

```python
from PIL import Image
import numpy as np

img_pil = Image.open("image.jpg")  # RGB by default
img_np = np.array(img_pil)         # shape: (H, W, C), dtype: uint8
```

#### b. NumPy → PIL

```python
img_pil = Image.fromarray(img_np)  # Assumes RGB format
```

---

### 2. **cv2 image ↔ PIL.Image**

#### a. cv2 (BGR) → PIL (RGB)

```python
import cv2
from PIL import Image

img_cv2 = cv2.imread("image.jpg")                   # BGR format
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert to RGB
img_pil = Image.fromarray(img_rgb)
```

#### b. PIL (RGB) → cv2 (BGR)

```python
img_np = np.array(img_pil)                          # RGB
img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)   # Convert to BGR
```

---

### 3. **NumPy (RGB) ↔ torch.Tensor (RGB)**

#### a. NumPy → Tensor

```python
import torch

img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # HWC → CHW
# Optional normalization to [0, 1]
img_tensor /= 255.0
```

#### b. Tensor → NumPy

```python
img_np = (img_tensor * 255).byte().permute(1, 2, 0).numpy()  # CHW → HWC
```

---

### 4. **PIL.Image ↔ torch.Tensor (RGB)**

#### a. PIL → Tensor

```python
from torchvision import transforms

transform = transforms.ToTensor()  # Converts to [0,1] and CHW
img_tensor = transform(img_pil)    # shape: (C, H, W), dtype: float32
```

#### b. Tensor → PIL

```python
from torchvision.transforms.functional import to_pil_image

img_pil = to_pil_image(img_tensor)  # Expects [0,1] float or [0,255] uint8
```

---

## ⚠️ Important Notes

1. **Color Format Matters**:

   * `cv2.imread()` returns BGR, but `PIL.Image.open()` returns RGB.
   * Always convert BGR ↔ RGB when switching between OpenCV and PIL.

2. **Data Types**:

   * `PIL.Image` works with `uint8` images.
   * `torch.Tensor` usually works in `float32`, normalized to \[0, 1].

3. **Tensor Shape**:

   * PyTorch expects `(C, H, W)` (channels first).
   * NumPy and PIL use `(H, W, C)` (channels last).

4. **Pixel Value Ranges**:

   * OpenCV and NumPy default to `[0, 255]`, `uint8`.
   * `transforms.ToTensor()` converts to `[0, 1]`, `float32`.

5. **Safe Conversion Advice**:

   * Always explicitly handle color space and data type during conversion.
   * When saving images with `cv2.imwrite`, ensure data is `uint8` and in BGR.
   * When using images in ML pipelines, normalize properly if needed (e.g., mean/std for pretrained models).
  
---

* ✅ Conversions between: `PIL.Image`, `np.ndarray`, `cv2`, `torch.Tensor`
* 📖 Reading methods for each format
* 💾 Saving methods for each format
* 🔧 Safe handling of color spaces, data types, and shape formats

---

## 📦 `image_utils.py`

```python
# image_utils.py

import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# ======================================================
# 🖼️ FORMAT CONVERSIONS
# ======================================================

# --- PIL <--> NumPy ---
def pil_to_numpy(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil)

def numpy_to_pil(img_np: np.ndarray) -> Image.Image:
    return Image.fromarray(img_np)

# --- OpenCV (BGR) <--> PIL (RGB) ---
def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    img_rgb = np.array(img_pil)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# --- NumPy <--> Torch Tensor ---
def numpy_to_tensor(img_np: np.ndarray, normalize: bool = True) -> torch.Tensor:
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    return tensor / 255.0 if normalize else tensor

def tensor_to_numpy(img_tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    if denormalize:
        img_tensor = img_tensor * 255.0
    return img_tensor.permute(1, 2, 0).byte().numpy()

# --- PIL <--> Torch Tensor ---
def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img_pil)

def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(img_tensor)

# --- OpenCV <--> Torch Tensor ---
def cv2_to_tensor(img_cv2: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(img_rgb)

def tensor_to_cv2(img_tensor: torch.Tensor) -> np.ndarray:
    img_rgb = tensor_to_numpy(img_tensor)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# ======================================================
# 📖 READ IMAGE METHODS
# ======================================================

def read_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def read_image_numpy(path: str) -> np.ndarray:
    img_pil = read_image_pil(path)
    return np.array(img_pil)

def read_image_cv2(path: str) -> np.ndarray:
    return cv2.imread(path)  # BGR format

def read_image_tensor(path: str) -> torch.Tensor:
    img_pil = read_image_pil(path)
    return transforms.ToTensor()(img_pil)  # (C, H, W), float32 [0,1]

# ======================================================
# 💾 SAVE IMAGE METHODS
# ======================================================

def save_image_pil(img_pil: Image.Image, path: str):
    img_pil.save(path)

def save_image_numpy(img_np: np.ndarray, path: str):
    img_pil = Image.fromarray(img_np)
    img_pil.save(path)

def save_image_cv2(img_cv2: np.ndarray, path: str):
    cv2.imwrite(path, img_cv2)  # Expects BGR format

def save_image_tensor(img_tensor: torch.Tensor, path: str):
    img_pil = to_pil_image(img_tensor)
    img_pil.save(path)
```

---

## ✅ Example Usage

```python
from image_utils import *

# Load from file
img_cv2 = read_image_cv2("image.jpg")         # OpenCV (BGR)
img_pil = read_image_pil("image.jpg")         # PIL (RGB)
img_np = read_image_numpy("image.jpg")        # NumPy (RGB)
img_tensor = read_image_tensor("image.jpg")   # Tensor (C,H,W), RGB

# Save to file
save_image_pil(img_pil, "out_pil.jpg")
save_image_numpy(img_np, "out_np.jpg")
save_image_cv2(img_cv2, "out_cv2.jpg")
save_image_tensor(img_tensor, "out_tensor.jpg")

# Convert between types
img_tensor = cv2_to_tensor(img_cv2)
img_cv2_back = tensor_to_cv2(img_tensor)
```

---

## ⚠️ Best Practices

* When saving with `cv2`, make sure the image is in BGR and `uint8`.
* When saving tensors, values must be in `[0, 1]` for `to_pil_image()` to work.
* `read_image_pil` uses `.convert("RGB")` to ensure consistent channel order.





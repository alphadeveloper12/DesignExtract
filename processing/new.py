import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    # Check if image has an alpha channel and remove it if necessary
    if im.shape[2] == 4:  # RGBA
        im = im[:, :, :3]  # Keep only the RGB channels

    # Convert to PyTorch tensor and normalize
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW format

    # Resize the image to the model input size
    im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=model_input_size, mode='bilinear',
                              align_corners=False).squeeze(0)

    # Normalize the image
    image = im_tensor / 255.0
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    image = normalize(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

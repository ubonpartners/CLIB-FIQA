import os
import torch
from model import clip
from model.models import convert_weights
import numpy as np
from utilities import *
import onnxruntime as ort
import torch
import torch.nn.functional as F
from itertools import product
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']
blur_list = ['hazy', 'blurry', 'clear']
occ_list = ['obstructed', 'unobstructed']
pose_list = ['profile', 'slight angle', 'frontal']
exp_list = ['exaggerated expression', 'typical expression']
ill_list = ['extreme lighting', 'normal lighting']

def img_tensor_batch(img_paths):
    """
    Given a list of image file paths, load and preprocess them into a single batched tensor.
    Output shape: [batch_size, 3, 224, 224]
    """
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))

    batch_tensor = torch.stack(images, dim=0)  # shape: [N, 3, 224, 224]
    return batch_tensor

def do_batch_onnx(session: ort.InferenceSession,
                  x: torch.Tensor):
    """
    Run the ONNX model on a batch of images + your fixed text prompts.
    Returns softmax’d (logits_per_image, logits_per_text) as torch.Tensor.
    """
    # 1) Move your torch inputs to CPU numpy
    #    ONNXRuntime expects numpy arrays
    img_np  = x.detach().cpu().numpy()

    # 2) Prepare the inputs dict — names must match input_names in export
    inputs = {
        "image": img_np
    }

    # 3) Run session; names must match output_names in export
    ort_outs = session.run(
        ["logits_per_image", "logits_per_text"],
        inputs
    )
    img_logits_np, _ = ort_outs

    # 4) Convert back to torch
    logits_per_image = torch.from_numpy(img_logits_np)

    # 5) (Optional) replicate your original softmax behavior
    #    Depending on where you applied softmax in PyTorch, you can skip this
    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image

if __name__ == "__main__":
    print("Load");
    session = ort.InferenceSession(
        "/mldata/fiqa/clib_fiqa_image_only2.onnx",
        providers=["CUDAExecutionProvider"]
    )
    print("Loaded ok")

    images=["./samples/1.jpg","./samples/2.jpg","./samples/3.jpg","./samples/4.jpg"
                       ,"./samples/5.jpg","./samples/6.jpg"]

    print("run\n");
    tensor_data=img_tensor_batch(images).cuda()
    logits_per_image = do_batch_onnx(session, tensor_data)
    logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
    logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
    quality_preds = dist_to_score(logits_quality).cpu().detach().numpy()
    print(quality_preds)
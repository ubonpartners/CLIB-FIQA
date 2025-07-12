import os
import torch
from model import clip
from model.models import convert_weights
import numpy as np
from utilities import *
import onnxruntime as ort
import torch.nn.functional as F
from itertools import product
from PIL import Image
import torchvision.transforms as T
import onnx
import onnx.helper
import onnxruntime as ort
import numpy as np


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

def patch_graph():
    model_path = "/mldata/fiqa/clib_fiqa.onnx"
    patched_path = "clib_fiqa_with_transpose2.onnx"

    # load
    model = onnx.load(model_path)
    graph = model.graph

    # find the node output name
    # usually node.output[0], but confirm by inspecting graph.node
    transpose_output_name = None
    for node in graph.node:
        if node.name == "/Transpose_2":
            transpose_output_name = node.output[0]
            break
    if transpose_output_name is None:
        raise RuntimeError("Could not find a node named 'Transpose_2' in the graph.")

    # create a new ValueInfoProto and append it to graph.output
    new_output = onnx.helper.make_empty_tensor_value_info(transpose_output_name)
    graph.output.append(new_output)

    # save patched model
    onnx.save(model, patched_path)
    print(f"Patched model saved to {patched_path}, exposed output: {transpose_output_name}")

def do_batch_onnx(session: ort.InferenceSession,
                  x: torch.Tensor,
                  text_tensor: torch.Tensor):
    """
    Run the ONNX model on a batch of images + fixed text prompts.
    Also extract the constant output of /Transpose_2 and save it.
    """
    img_np = x.detach().cpu().numpy()
    txt_np = text_tensor.detach().cpu().numpy()
    inputs = {"image": img_np, "text": txt_np}

    # 1) Extract the constant tensor at /Transpose_2
    transpose_const = session.run(
        ["/Transpose_2_output_0"],   # change here to your node’s output name
        inputs
    )[0]
    # Save it out
    np.save("transpose2_constant.npy", transpose_const)
    print(f"Saved /Transpose_2 constant (shape {transpose_const.shape}) to transpose2_constant.npy")
    # Saved /Transpose_2 constant (shape (1024, 360)) to transpose2_constant.npy
    exit()

    # 2) Now do your normal forward for logits
    ort_outs = session.run(
        ["logits_per_image", "logits_per_text"],
        inputs
    )
    img_logits_np, _ = ort_outs

    logits_per_image = torch.from_numpy(img_logits_np)
    logits_per_image = F.softmax(logits_per_image, dim=1)
    return logits_per_image

if __name__ == "__main__":
    print("Load")
    #patch_graph();
    #exit()


    session = ort.InferenceSession(
        "clib_fiqa_with_transpose2.onnx",
        providers=["CUDAExecutionProvider"]
    )
    print("Loaded ok")

    # Load your text embeddings once
    joint_texts = torch.from_numpy(
        np.load("/mldata/fiqa/joint_texts.npy")
    ).cuda()

    # Prepare images
    images = [
        "./samples/1.jpg","./samples/2.jpg","./samples/3.jpg",
        "./samples/4.jpg","./samples/5.jpg","./samples/6.jpg"
    ]
    tensor_data = img_tensor_batch(images).cuda()

    # Run inference (this will also save the constant)
    logits_per_image = do_batch_onnx(session, tensor_data, joint_texts)

    # Post-processing as before
    logits_per_image = logits_per_image.view(
        -1,
        len(blur_list),
        len(occ_list),
        len(pose_list),
        len(exp_list),
        len(ill_list),
        len(quality_list)
    )
    logits_quality  = logits_per_image.sum((1,2,3,4,5))
    quality_preds   = dist_to_score(logits_quality).cpu().detach().numpy()
    print(quality_preds)
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
from onnxconverter_common import float16
import onnx

quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']
blur_list = ['hazy', 'blurry', 'clear']
occ_list = ['obstructed', 'unobstructed']
pose_list = ['profile', 'slight angle', 'frontal']
exp_list = ['exaggerated expression', 'typical expression']
ill_list = ['extreme lighting', 'normal lighting']
joint_texts = torch.cat([clip.tokenize(f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {q} quality")
                for b, o, p, e, l, q in product(blur_list, occ_list, pose_list, exp_list, ill_list, quality_list)]).cuda()

pose_map = {0:pose_list[0], 1:pose_list[1], 2:pose_list[2]}
blur_map = {0:blur_list[0], 1:blur_list[1], 2:blur_list[2]}
occ_map  = {0:occ_list[0],  1:occ_list[1]}
ill_map  = {0:ill_list[0],  1:ill_list[1]}
exp_map =  {0:exp_list[0],  1:exp_list[1]}

def img_tensor(imgPath):
    img = Image.open(imgPath).convert("RGB")
    transform = T.Compose([
                T.Resize([224, 224]),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ])
    img_tensor = transform(img)
    data = img_tensor.unsqueeze(dim=0)
    return data

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

def backboneSet(clip_model):
    net, _ = clip.load(clip_model, device='cuda', jit=False)
    return net

@torch.no_grad()
def do_batch(model, x, text):
    batch_size = x.size(0)
    x = x.view(-1, x.size(1), x.size(2), x.size(3))
    logits_per_image, logits_per_text = model.forward(x, text)
    logits_per_image = logits_per_image.view(batch_size, -1)
    logits_per_text = logits_per_text.view(-1, batch_size)
    logits_per_image = F.softmax(logits_per_image, dim=1)
    logits_per_text = F.softmax(logits_per_text, dim=1)
    return logits_per_image, logits_per_text

def do_batch_onnx(session: ort.InferenceSession,
                  x: torch.Tensor,
                  text_tensor: torch.Tensor):
    """
    Run the ONNX model on a batch of images + your fixed text prompts.
    Returns softmax’d (logits_per_image, logits_per_text) as torch.Tensor.
    """
    # 1) Move your torch inputs to CPU numpy
    #    ONNXRuntime expects numpy arrays
    img_np  = x.detach().cpu().numpy()
    txt_np  = text_tensor.detach().cpu().numpy()

    # 2) Prepare the inputs dict — names must match input_names in export
    inputs = {
        "image": img_np,
        "text":  txt_np,
    }

    # 3) Run session; names must match output_names in export
    ort_outs = session.run(
        ["logits_per_image", "logits_per_text"],
        inputs
    )
    img_logits_np, txt_logits_np = ort_outs

    # 4) Convert back to torch
    logits_per_image = torch.from_numpy(img_logits_np)
    logits_per_text  = torch.from_numpy(txt_logits_np)

    # 5) (Optional) replicate your original softmax behavior
    #    Depending on where you applied softmax in PyTorch, you can skip this
    logits_per_image = F.softmax(logits_per_image, dim=1)
    logits_per_text  = F.softmax(logits_per_text, dim=0)

    return logits_per_image, logits_per_text

def export_onnx(model, text_tensor, output_path="model.onnx"):
    """
    Exports the CLIP-based quality assessment model to ONNX format.
    Args:
        model: the loaded CLIP model
        text_tensor: the tokenized text tensor (joint_texts)
        output_path: where to save the ONNX model
    """
    model = model.float()
    model.eval()
    # Create a dummy image input (batch size 1)
    dummy_image = torch.randn(1, 3, 224, 224, device='cuda')
    # Export the model: inputs are image and text tensors
    torch.onnx.export(
        model,
        (dummy_image, text_tensor),
        output_path,
        input_names=["image", "text"],
        output_names=["logits_per_image", "logits_per_text"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits_per_image": {0: "batch_size"},
            "logits_per_text": {1: "batch_size"}
        },
        opset_version=14
    )
    print(f"ONNX model exported to {output_path}")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clip_model = "./weights/RN50.pt"
    clip_weights = "./weights/CLIB-FIQA_R50.pth"
    image_path = "./samples/1.jpg"

    model = backboneSet(clip_model)
    model = load_net_param(model, clip_weights)

    #export_onnx(model, joint_texts)
    #np.save("joint_texts.npy", joint_texts.detach().cpu().numpy())

    model_fp32 = onnx.load("model.onnx")
    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    # Save new model
    onnx.save(model_fp16, "model_fp16.onnx")

    session = ort.InferenceSession(
        "./model.onnx",
        providers=["CUDAExecutionProvider"]
    )

    images=["./samples/1.jpg","./samples/2.jpg","./samples/3.jpg","./samples/4.jpg"
                       ,"./samples/5.jpg","./samples/6.jpg"]

    for image_path in images:
        tensor_data = img_tensor(image_path).cuda()
        for i in range(2):
            if (i==0):
                logits_per_image, _, = do_batch(model, tensor_data, joint_texts)
            else:
                logits_per_image, _, = do_batch_onnx(session, tensor_data, joint_texts)
            logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
            logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
            logits_blur     = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(2), dim=1)[1].cpu().detach().numpy().squeeze(0)
            logits_occ      = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
            logits_pose     = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
            logits_exp      = torch.max(logits_per_image.sum(6).sum(5).sum(3).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
            logits_ill      = torch.max(logits_per_image.sum(6).sum(4).sum(3).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
            print(logits_quality.shape)
            quality_preds = dist_to_score(logits_quality).cpu().detach().numpy().squeeze(0)

            output_msg = f"a photo of a [{blur_map[int(logits_blur)]}], [{occ_map[int(logits_occ)]}], and [{pose_map[int(logits_pose)]}] face with [{exp_map[int(logits_exp)]}] under [{ill_map[int(logits_ill)]}]"
            print(output_msg)
            print(f"quality prediction = {quality_preds}")

    tensor_data=img_tensor_batch(images).cuda()
    logits_per_image, _, = do_batch_onnx(session, tensor_data, joint_texts)
    logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
    logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
    quality_preds = dist_to_score(logits_quality).cpu().detach().numpy()
    print(quality_preds)
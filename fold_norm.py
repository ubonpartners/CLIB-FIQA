import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

def fold_norm_keep_input(in_path: str,
                         out_path: str,
                         input_name: str = "image",
                         norm_output: str = "normalized_image",
                         mean: list = [0.48145466, 0.4578275, 0.40821073],
                         std:  list = [0.26862954, 0.26130258, 0.27577711]):
    # 1) Load
    model = onnx.load(in_path)
    graph = model.graph

    # 2) Create mean/std initializers (shape [1,3,1,1])
    mean_arr = numpy_helper.from_array(
        np.array(mean, dtype=np.float32).reshape((1,3,1,1)),
        name="Norm_Mean"
    )
    std_arr = numpy_helper.from_array(
        np.array(std, dtype=np.float32).reshape((1,3,1,1)),
        name="Norm_Std"
    )
    graph.initializer.extend([mean_arr, std_arr])

    # 3) Insert Sub and Div nodes at the front:
    #    Sub(image, Norm_Mean) -> tmp_norm
    #    Div(tmp_norm, Norm_Std) -> normalized_image
    sub_node = helper.make_node(
        "Sub",
        inputs=[input_name, "Norm_Mean"],
        outputs=["tmp_norm"],
        name="FoldNorm_Sub"
    )
    div_node = helper.make_node(
        "Div",
        inputs=["tmp_norm", "Norm_Std"],
        outputs=[norm_output],
        name="FoldNorm_Div"
    )
    # Prepend them so they run before everything else
    graph.node.insert(0, div_node)
    graph.node.insert(0, sub_node)

    # 4) Re-wire every node that formerly consumed 'image'
    #    to now consume 'normalized_image' instead.
    for node in graph.node[2:]:  # skip our two new nodes
        node.input[:] = [
            norm_output if inp == input_name else inp
            for inp in node.input
        ]

    # 5) (Optional) update any value_info entries
    for vi in graph.value_info:
        if vi.name == input_name:
            # declare that 'normalized_image' exists
            graph.value_info.extend([
                helper.make_tensor_value_info(
                    norm_output,
                    TensorProto.FLOAT,
                    # unknown shape: omit dims or mirror image shape
                )
            ])
            break

    # 6) Save
    onnx.save(model, out_path)
    print(f"Saved model with folded normalization to {out_path}")

if __name__ == "__main__":
    fold_norm_keep_input(
        in_path  = "/mldata/fiqa/clib_fiqa_image_only.onnx",
        out_path = "/mldata/fiqa/clib_fiqa_norm.onnx"
    )

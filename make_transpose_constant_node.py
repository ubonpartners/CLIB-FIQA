import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto

# Paths
model_path = "/mldata/fiqa/clib_fiqa.onnx"
constant_npy = "transpose2_constant.npy"
output_path = "/mldata/fiqa/clib_fiqa_patched.onnx"

# 1. Load model and constant value
model = onnx.load(model_path)
graph = model.graph
const_val = np.load(constant_npy).astype(np.float32)  # make sure dtype matches

# Determine the name of the Transpose_2 output
# You told us it's "/Transpose_2_output_0"
old_output_name = "/Transpose_2_output_0"

# 2. Create a new initializer tensor for the constant
const_tensor_name = "Transpose_2_Constant"
const_initializer = numpy_helper.from_array(const_val, name=const_tensor_name)

# 3. Create a Constant node that emits that tensor
const_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=[old_output_name],
    name="Transpose2_Replacement_Const",
    value=const_initializer  # embed the tensor here
)

# 4. Remove the old Transpose node
new_nodes = []
for node in graph.node:
    # match by node.name or by op_type + output
    if node.name == "Transpose_2" or old_output_name in node.output:
        # skip this node entirely
        continue
    new_nodes.append(node)

# 5. Insert the new Constant node in roughly the same position
#    (here, we'll put it at the start of the graph.node list;
#     you could also insert at the original index if you preserved it)
graph.ClearField('node')
graph.node.extend([const_node] + new_nodes)

# 6. Add the initializer to the graph (so external tools know about it)
graph.initializer.append(const_initializer)

# 7. (Optional) clean up any dangling inputs
#    If '/Transpose_2_input_*' names are no longer used, you can remove them from graph.input

# 8. Save patched model
onnx.save(model, output_path)
print(f"Patched model saved to {output_path}")
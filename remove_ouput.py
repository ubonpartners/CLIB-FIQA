import onnx
from onnx import helper

def remove_output_and_prune(in_path: str,
                            out_path: str,
                            drop_output: str,
                            keep_input_names: list):
    # 1. Load
    model = onnx.load(in_path)
    graph = model.graph

    # 2. Remove the unwanted output via ClearField + extend
    kept_outputs = [o for o in graph.output if o.name != drop_output]
    graph.ClearField('output')
    graph.output.extend(kept_outputs)

    # 3. Build “needed” set (remaining outputs + inputs you want)
    needed = set(o.name for o in graph.output)
    for inp in graph.input:
        if inp.name in keep_input_names:
            needed.add(inp.name)

    # 4. Back-traverse producers
    producer = {out: node for node in graph.node for out in node.output}
    stack = list(needed)
    while stack:
        name = stack.pop()
        if name in producer:
            node = producer[name]
            for inp in node.input:
                if inp not in needed:
                    needed.add(inp)
                    stack.append(inp)

    # 5. Prune nodes
    kept_nodes = [n for n in graph.node if any(o in needed for o in n.output)]
    graph.ClearField('node')
    graph.node.extend(kept_nodes)

    # 6. Prune inputs
    kept_inputs = [i for i in graph.input if i.name in keep_input_names and i.name in needed]
    graph.ClearField('input')
    graph.input.extend(kept_inputs)

    # 7. Prune initializers
    kept_inits = [init for init in graph.initializer if init.name in needed]
    graph.ClearField('initializer')
    graph.initializer.extend(kept_inits)

    # 8. (Optional) drop any leftover value_info
    graph.ClearField('value_info')

    # 9. Save
    onnx.save(model, out_path)
    print(f"Saved model without '{drop_output}' to {out_path}")



if __name__ == "__main__":
    remove_output_and_prune(
        in_path="/mldata/fiqa/clib_fiqa_image_only.onnx",
        out_path="clib_fiqa_final.onnx",
        drop_output="logits_per_text",
        keep_input_names=["image"]
    )
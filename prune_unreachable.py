import onnx
from onnx import helper, numpy_helper

def prune_onnx_model(in_path: str, out_path: str,
                     keep_output_names: list,
                     keep_input_names: list):
    # 1. Load
    model = onnx.load(in_path)
    graph = model.graph

    # 2. Backwards traversal from outputs → collect all needed value names
    needed = set(keep_output_names)
    # value producers: map each value_name → the node that produces it
    producer = {out: node for node in graph.node for out in node.output}

    # walk backwards
    stack = list(keep_output_names)
    while stack:
        val = stack.pop()
        if val in producer:
            node = producer[val]
            # mark this node and all its inputs as needed
            for inp in node.input:
                if inp not in needed:
                    needed.add(inp)
                    stack.append(inp)

    # 3. Filter nodes: keep only those that produce any needed output
    new_nodes = [n for n in graph.node if any(o in needed for o in n.output)]

    print(f"Original had {len(graph.node)} new has {len(new_nodes)}")

    graph.ClearField('node')
    graph.node.extend(new_nodes)

    # 4. Filter inputs: only those in keep_input_names *and* that actually appear
    new_inputs = [i for i in graph.input
                  if (i.name in keep_input_names and i.name in needed)]
    graph.ClearField('input')
    graph.input.extend(new_inputs)

    # 5. (Optional) Remove any unused value_info / initializers
    graph.ClearField('value_info')
    # you can similarly prune graph.initializer if you like

    # 6. Save
    onnx.save(model, out_path)
    print(f"Pruned model saved to {out_path}")

if __name__ == "__main__":
    prune_onnx_model(
        in_path   = "clib_fiqa_patched.onnx",
        out_path  = "clib_fiqa_pruned.onnx",
        keep_output_names = ["logits_per_image", "logits_per_text"],
        keep_input_names  = ["image"]              # drop "text"
    )
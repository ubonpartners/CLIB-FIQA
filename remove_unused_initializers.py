import onnx

def remove_unused_initializers(model_path: str, output_path: str):
    model = onnx.load(model_path)
    graph = model.graph

    # 1. Collect every “used” name from node inputs and from graph inputs
    used = set()
    for node in graph.node:
        for inp in node.input:
            used.add(inp)
    for inp in graph.input:
        used.add(inp.name)

    # 2. Filter the initializer list
    kept_initializers = [
        init for init in graph.initializer
        if init.name in used
    ]
    removed = set(init.name for init in graph.initializer) - set(init.name for init in kept_initializers)

    # 3. Replace graph.initializer
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    # 4. (Optional) log what you removed
    print(f"Removed unused initializers ({len(removed)}):")
    for name in sorted(removed):
        print("  -", name)

    # 5. Save
    onnx.save(model, output_path)
    print(f"Cleaned model saved to {output_path}")

if __name__ == "__main__":
    remove_unused_initializers(
        "/mldata/fiqa/clib_fiqa_image_only.onnx",
        "/mldata/fiqa/clib_fiqa_image_only2.onnx"
    )
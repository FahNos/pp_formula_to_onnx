import onnx
import numpy as np
from onnx import helper, TensorProto
import os

def _fix_graph_recursively(graph):
    fixes_count = 0
    
    for init in graph.initializer:
        if init.data_type == TensorProto.DOUBLE:
            if init.HasField("raw_data"):
                data = np.frombuffer(init.raw_data, dtype=np.float64)
            else:
                data = np.array(init.double_data, dtype=np.float64)
            data_float32 = data.astype(np.float32)
            init.data_type = TensorProto.FLOAT
            init.ClearField("double_data")
            init.raw_data = data_float32.tobytes()
            fixes_count += 1
            
    for node in graph.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    fixes_count += 1
        elif node.op_type == 'Constant':
             for attr in node.attribute:
                if attr.name == 'value':
                    tensor = attr.t
                    if tensor.data_type == TensorProto.DOUBLE:
                        if tensor.HasField("raw_data"):
                            data = np.frombuffer(tensor.raw_data, dtype=np.float64)
                        else:
                            data = np.array(tensor.double_data, dtype=np.float64)
                        data_float32 = data.astype(np.float32)
                        new_tensor = onnx.helper.make_tensor(
                            name=tensor.name, data_type=TensorProto.FLOAT,
                            dims=tensor.dims, vals=data_float32.tobytes(), raw=True
                        )
                        attr.t.CopyFrom(new_tensor)
                        fixes_count += 1
                        
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                fixes_count += _fix_graph_recursively(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for i, subgraph in enumerate(attr.graphs):
                    fixes_count += _fix_graph_recursively(subgraph)

    all_tensor_definitions = list(graph.input) + list(graph.output) + list(graph.value_info)
    for tensor in all_tensor_definitions:
        if tensor.type.tensor_type.elem_type == TensorProto.DOUBLE:
            tensor.type.tensor_type.elem_type = TensorProto.FLOAT
            fixes_count += 1
            
    return fixes_count

def fix_onnx_model_recursively(onnx_path, output_path):

    print(f"\n=== BẮT ĐẦU SỬA LỖI ĐỆ QUY: {onnx_path} ===")
    if not os.path.exists(onnx_path):
        print(f"LỖI: Không tìm thấy file tại {onnx_path}")
        return False

    try:
        model = onnx.load(onnx_path)
        
        total_fixes = _fix_graph_recursively(model.graph)
        
        print(f"\nTổng số lỗi đã sửa trên toàn bộ các đồ thị: {total_fixes}")

        print(f"Đang lưu model đã sửa tới: {output_path}")
        onnx.save(model, output_path)

        print("Đang xác thực model đã sửa...")
        onnx.checker.check_model(output_path)
        print("Xác thực THÀNH CÔNG!")
        return True

    except Exception as e:
        print(f"Một lỗi đã xảy ra trong quá trình sửa lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixed_model(fixed_onnx_path):
    print(f"\n=== KIỂM TRA LẠI MODEL ĐÃ SỬA: {fixed_onnx_path} ===")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(fixed_onnx_path)
        print("Model đã được tải thành công vào ONNX Runtime!")
        return True
    except Exception as e:
        print(f"Không thể tải model đã sửa vào ONNX Runtime.")
        print(f"   Lỗi: {e}")
        return False

if __name__ == '__main__':

    original_head_path = "./output/onnx_models/head.onnx"    
    fixed_head_path = "./output/onnx_models/head_fixed.onnx"
    
    fix_success = fix_onnx_model_recursively(original_head_path, fixed_head_path)
    
    if fix_success:
        test_fixed_model(fixed_head_path)

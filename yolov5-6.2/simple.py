import onnxsim
import onnx

filepath = "weights/yolov5.onnx"
output_path = "weights/yolov5_opt.onnx"
onnx_model = onnx.load(filepath)  # load onnx model
model_simp, check = onnxsim.simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')
# model_onnx, check = onnxsim.simplify(f)

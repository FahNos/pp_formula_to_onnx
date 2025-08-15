from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import tools.program as program
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.utility import get_image_file_list

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not installed. Please install it with: pip install onnxruntime")


class ONNXInferenceEngine:
   
    def __init__(self, backbone_onnx_path=None, head_onnx_path=None):
        self.ort = ort
        ort.set_default_logger_severity(4)
        self.backbone_session = None
        self.head_session = None
        
        if backbone_onnx_path and os.path.exists(backbone_onnx_path):
            try:
                self.backbone_session = ort.InferenceSession(backbone_onnx_path)
                logger.info(f"Loaded backbone ONNX model from: {backbone_onnx_path}")
                logger.info("Backbone inputs:")
                for input_meta in self.backbone_session.get_inputs():
                    logger.info(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")
            except Exception as e:
                logger.error(f"Failed to load backbone ONNX model: {e}")
        
        if head_onnx_path and os.path.exists(head_onnx_path):
            try:
                self.head_session = ort.InferenceSession(head_onnx_path)
                logger.info(f"Loaded head ONNX model from: {head_onnx_path}")
                logger.info("Head inputs:")
                for input_meta in self.head_session.get_inputs():
                    logger.info(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")
            except Exception as e:
                logger.error(f"Failed to load head ONNX model: {e}")
    
    def run_backbone(self, image_array):
        if self.backbone_session is None:
            raise RuntimeError("Backbone ONNX model not loaded or failed to load.")
        
        input_name = self.backbone_session.get_inputs()[0].name
        outputs = self.backbone_session.run(None, {input_name: image_array})
        return outputs[0]
    
    def run_head_single_step(self, decoder_input_ids, encoder_outputs):
        if self.head_session is None:
            raise RuntimeError("Head ONNX model not loaded or failed to load.")
        
        input_names = [input_meta.name for input_meta in self.head_session.get_inputs()]
        
        feed_dict = {}
        for name in input_names:
            if 'decoder' in name.lower() or 'input_ids' in name.lower():
                feed_dict[name] = decoder_input_ids
            elif 'encoder' in name.lower() or 'outputs' in name.lower():
                feed_dict[name] = encoder_outputs
        
        outputs = self.head_session.run(None, feed_dict)
        return outputs[0]
    
    def generate_with_onnx(self, encoder_outputs, max_length=1536, eos_token_id=2, 
                          decoder_start_token_id=0, temperature=0.2, do_sample=False, top_p=0.95):
        if self.head_session is None:
            raise RuntimeError("Head ONNX model not loaded or failed to load.")
        
        batch_size = encoder_outputs.shape[0]
        generated_ids = np.array([[decoder_start_token_id]] * batch_size, dtype=np.int64)
        
        for _ in range(max_length):
            logits = self.run_head_single_step(generated_ids, encoder_outputs)            
           
            next_token = np.argmax(logits, axis=-1).astype(np.int64)
            
            next_token = next_token.reshape(-1, 1)
            generated_ids = np.concatenate([generated_ids, next_token], axis=-1)
            
            if np.all(next_token.flatten() == eos_token_id):
                break
        
        return generated_ids


def run_onnx_inference_pipeline(images, onnx_engine, post_process_class, max_length=1536):
   
    if hasattr(images, 'numpy'):
        image_array = images.numpy()
    else:
        image_array = images
    
    image_array = image_array.astype(np.float32)

    if image_array.shape[1] == 1:
        logger.info("Input image has 1 channel. Converting to 3 channels for ONNX model...")
        image_array = np.repeat(image_array, 3, axis=1)

    logger.info(f"Input image shape: {image_array.shape}")
    
    if onnx_engine.backbone_session:
        logger.info("Running backbone ONNX inference...")
        encoder_outputs = onnx_engine.run_backbone(image_array)
        logger.info(f"Encoder outputs shape: {encoder_outputs.shape}")
    else:
        logger.warning("No backbone ONNX model loaded. Skipping backbone inference.")
        encoder_outputs = image_array

    if onnx_engine.head_session:
        logger.info("Running head ONNX generation...")
        predictions = onnx_engine.generate_with_onnx(
            encoder_outputs, 
            max_length=max_length
        )
        logger.info(f"Generated sequence shape: {predictions.shape}")
        
        predictions_tensor = paddle.to_tensor(predictions)
        
        post_result = post_process_class(predictions_tensor)
        return predictions, post_result
    else:
        logger.error("No head ONNX model loaded. Cannot generate results.")
        return None, "HEAD_MODEL_FAILED"


def main():    
    global_config = config["Global"]

    logger.info("Initializing post-processor and image transformers...")
    
    post_process_class = build_post_process(config["PostProcess"], global_config)
    
    # preprocess
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    ops = create_operators(transforms, global_config)

    backbone_onnx_path = global_config.get("backbone_onnx_path", "./onnx_model/backbone.onnx")
    head_onnx_path = global_config.get("head_onnx_path", "./onnx_model/head.onnx")
    infer_imgs = global_config.get("infer_img", "./inference_images/")
    save_res_path = global_config.get("save_res_path", "./output/rec/onnx_predicts.txt")
    max_length = global_config.get("max_new_tokens", 1536)

    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    logger.info("=== Initializing ONNX Inference Engine ===")
    onnx_engine = ONNXInferenceEngine(backbone_onnx_path, head_onnx_path)

    logger.info("=== Starting ONNX Inference ===")
    
    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(infer_imgs):
            logger.info(f"Inferring image: {file}")

            try:
                # image prepocess
                with open(file, "rb") as f:
                    img = f.read()
                    data = {"image": img, "filename": file}
                batch = transform(data, ops)
                images = np.expand_dims(batch[0], axis=0)
                
                _, post_result = run_onnx_inference_pipeline(
                    images, onnx_engine, post_process_class, max_length=max_length
                )
                
                result_text = str(post_result[0]) if post_result else "Prediction failed"
                logger.info(f"\t Result: {result_text}")
                
                fout.write(f"{os.path.basename(file)}\t{result_text}\n")
            
            except Exception as e:
                logger.error(f"Error while processing {file}: {e}")
                fout.write(f"{os.path.basename(file)}\tERROR\n")
    
    logger.info(f"=== Inference completed. Results saved to: {save_res_path} ===")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
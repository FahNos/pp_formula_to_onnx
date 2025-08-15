from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import json
import paddle.nn as nn


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
from ppocr.modeling.heads.rec_unimernet_head import (
    MBartForCausalLM,
    MBartDecoder,
    MBartConfig,
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    zeros_,
    ones_,
    kaiming_normal_,
    trunc_normal_,
    xavier_uniform_,
    CausalLMOutputWithCrossAttentions,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    UniMERNetHead,
)


def patch_attention_mask_for_export():
    import ppocr.modeling.heads.rec_unimernet_head as unimernet_module
    
    original_to_4d = unimernet_module.AttentionMaskConverter.to_4d
    original_expand_mask = unimernet_module.AttentionMaskConverter._expand_mask
    original_make_causal_mask = unimernet_module.AttentionMaskConverter._make_causal_mask
    
    unimernet_module.AttentionMaskConverter.to_4d = to_4d_safe_for_export
    unimernet_module.AttentionMaskConverter._expand_mask = expand_mask_safe_for_export
    unimernet_module.AttentionMaskConverter._make_causal_mask = make_causal_mask_safe_for_export
    
    return original_to_4d, original_expand_mask, original_make_causal_mask

def to_4d_safe_for_export(self, attention_mask_2d, query_length, dtype, key_value_length, is_export=False):
    if is_export:
        return self.to_4d_safe_export(attention_mask_2d, query_length, dtype, key_value_length)
    else:
        input_shape = (attention_mask_2d.shape[0], query_length)
        
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1])
        
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                key_value_length = query_length
            
            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape, dtype, past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window, is_export=False,
            )
            
            if hasattr(causal_4d_mask, 'masked_fill_'):
                expanded_attn_mask = causal_4d_mask.masked_fill_(
                    expanded_attn_mask.cast(paddle.bool), paddle.finfo(dtype).min
                )
            else:
                mask_bool = expanded_attn_mask.cast(paddle.bool)
                min_val = paddle.finfo(dtype).min
                expanded_attn_mask = paddle.where(mask_bool, min_val, causal_4d_mask)
        
        return expanded_attn_mask

def expand_mask_safe_for_export(self, mask, dtype, tgt_len=None):
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    
    expanded_mask = mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)
    inverted_mask = 1.0 - expanded_mask
    
    mask_bool = inverted_mask.cast(paddle.bool)
    min_val = paddle.finfo(dtype).min
    return paddle.where(mask_bool, min_val, inverted_mask)

def make_causal_mask_safe_for_export(input_ids_shape, dtype, past_key_values_length=0, sliding_window=None, is_export=False):
    bsz, tgt_len = input_ids_shape
    
    mask = paddle.full((tgt_len, tgt_len), paddle.finfo(dtype).min, dtype=dtype)
    mask_cond = paddle.arange(mask.shape[-1])
    
    condition = mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1])
    mask = paddle.where(condition, paddle.to_tensor(0.0, dtype=dtype), mask)
    
    return mask[None, None, :, :].expand([bsz, 1, tgt_len, tgt_len + past_key_values_length])

def validate_model_dtypes(model, model_name="Model"): 
    dtype_stats = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in dtype_stats:
            dtype_stats[dtype_str] = []
        dtype_stats[dtype_str].append(name)
    
    logger.info(f"{model_name} parameter dtypes:")
    for dtype, param_names in dtype_stats.items():
        logger.info(f"  {dtype}: {len(param_names)} parameters")
        if len(param_names) <= 5:
            logger.info(f"    {param_names}")
        else:
            logger.info(f"    {param_names[:3]} ... (and {len(param_names)-3} more)")
    
    return dtype_stats

def deep_convert_to_float32(model):   
    logger.info("Performing deep conversion to float32...")
    
    with paddle.no_grad():
        for name, param in model.named_parameters():
            if param.dtype != paddle.float32:
                param.set_value(param.cast(paddle.float32))
                logger.debug(f"Converted param {name}: -> float32")
        
        for name, buffer in model.named_buffers():
            if buffer.dtype != paddle.float32:
                model._buffers[name] = buffer.cast(paddle.float32)
                logger.debug(f"Converted buffer {name}: -> float32")
    
    model.eval()
    
    for name, module in model.named_sublayers():
        if hasattr(module, '_parameters'):
            for param_name, param in module._parameters.items():
                if param is not None and param.dtype != paddle.float32:
                    module._parameters[param_name] = param.cast(paddle.float32)
                    logger.debug(f"Converted nested param {name}.{param_name}: -> float32")
        
        if hasattr(module, '_buffers'):
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None and buffer.dtype != paddle.float32:
                    module._buffers[buffer_name] = buffer.cast(paddle.float32)
                    logger.debug(f"Converted nested buffer {name}.{buffer_name}: -> float32")
    
    logger.info("Deep conversion completed")
    return model

def force_model_float32_state(model):
    model.eval()
    
    paddle.set_default_dtype('float32')
    
    with paddle.no_grad():
        for name, module in model.named_sublayers():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dtype != paddle.float32:
                    new_weight = module.weight.cast(paddle.float32)
                    module.weight.set_value(new_weight)
            
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype != paddle.float32:
                    new_bias = module.bias.cast(paddle.float32)
                    module.bias.set_value(new_bias)
    
    return model

class PPFormulaNet_Head_SingleStep(UniMERNetHead):

    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        decoder_layers=8,
        encoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_hidden_size=1024,
        is_export=False,
        length_aware=True,
        use_parallel=False,
        parallel_step=3,
    ):
        super().__init__()

        mbart_config_dict = {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "add_cross_attention": True,
            "add_final_layer_norm": True,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "d_model": decoder_hidden_size,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": decoder_ffn_dim,
            "decoder_layerdrop": 0.0,
            "decoder_layers": decoder_layers,
            "dropout": 0.1,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "eos_token_id": 2,
            "forced_eos_token_id": 2,
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "output_hidden_states": False,
            "max_position_embeddings": max_new_tokens,
            "model_type": "mbart",
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "scale_embedding": True,
            "tie_word_embeddings": False,
            "transformers_version": "4.40.0",
            "use_cache": True,
            "use_return_dict": True,
            "vocab_size": 50000,
            "_attn_implementation": "eager",
            "hidden_size": decoder_hidden_size,
            "use_parallel": False,  # Force single step mode
            "parallel_step": 1,     # Single step only
            "is_export": is_export,
        }

        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.is_export = is_export
        self.max_seq_len = max_new_tokens
        self.config_decoder = MBartConfig(**mbart_config_dict)
        self.encoder_hidden_size = encoder_hidden_size

        self.decoder = MBartForCausalLM(self.config_decoder)
        
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_hidden_size, self.config_decoder.hidden_size
            )

        self.eos_token_id = 2
        self.pad_token_id = self.config_decoder.pad_token_id

        generation_config = {
            "max_length": max_new_tokens + 1,
            "forced_eos_token_id": 2,
        }
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config["max_length"],
                generation_config["forced_eos_token_id"],
            )
        )

    def single_step_forward(
        self,
        decoder_input_ids,
        encoder_outputs,
        past_key_values=None,
        decoder_attention_mask=None,
        use_cache=True,
    ):       
        batch_size = decoder_input_ids.shape[0]
        
        if decoder_attention_mask is None:
            decoder_attention_mask = paddle.ones_like(decoder_input_ids)

        if isinstance(encoder_outputs, (list, tuple)):
            encoder_hidden_states = encoder_outputs[0].last_hidden_state
        else:
            encoder_hidden_states = encoder_outputs
    
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        with paddle.no_grad():
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                inputs_embeds=None,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=True,
            )

        logits = decoder_outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        processed_logits = self.logits_processor(decoder_input_ids, logits.unsqueeze(1))
        processed_logits = processed_logits.squeeze(1)  # [batch_size, vocab_size]
        
        if self.do_sample:
            if self.temperature != 1.0:
                processed_logits = processed_logits / self.temperature
            
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = paddle.sort(processed_logits, descending=True)
                cumulative_probs = paddle.cumsum(paddle.nn.functional.softmax(sorted_logits, axis=-1), axis=-1)
                
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                processed_logits = processed_logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = paddle.nn.functional.softmax(processed_logits, axis=-1)
            next_token = paddle.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = paddle.argmax(processed_logits, axis=-1)

        is_finished = (next_token == self.eos_token_id)

        return {
            'logits': logits,
            'processed_logits': processed_logits,
            'past_key_values': decoder_outputs.past_key_values,
            'next_token': next_token,
            'is_finished': is_finished,
        }

def convert_head_to_single_step(original_head):
 
    single_step_head = PPFormulaNet_Head_SingleStep(
        max_new_tokens=original_head.max_seq_len,
        decoder_start_token_id=original_head.decoder_start_token_id,
        temperature=original_head.temperature,
        do_sample=original_head.do_sample,
        top_p=original_head.top_p,
        decoder_layers=original_head.config_decoder.decoder_layers,
        encoder_hidden_size=original_head.encoder_hidden_size,
        decoder_ffn_dim=original_head.config_decoder.decoder_ffn_dim,
        decoder_hidden_size=original_head.config_decoder.hidden_size,
        is_export=original_head.is_export,
    )
    
    single_step_head.decoder.set_state_dict(original_head.decoder.state_dict())
    
    if hasattr(original_head, 'enc_to_dec_proj') and hasattr(single_step_head, 'enc_to_dec_proj'):
        single_step_head.enc_to_dec_proj.set_state_dict(original_head.enc_to_dec_proj.state_dict())
    
    single_step_head.eval()
    logger.info("Successfully converted head to single step mode")
    
    return single_step_head


class StaticBackboneWrapper(nn.Layer):
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
    
    def forward(self, x):
        return self.backbone(x)

def export_to_static_graph(backbone_model, single_step_head, save_dir="./output/static_models"):
    import paddle
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    originals = patch_attention_mask_for_export()
    
    backbone_static_path = None
    head_static_path = None
    
    try:
        logger.info("=== EXPORTING BACKBONE TO STATIC GRAPH ===")
        if backbone_model is not None:
            static_backbone = StaticBackboneWrapper(backbone_model)
            static_backbone.eval()
            
            input_spec = paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32', name='image')
            static_backbone = paddle.jit.to_static(static_backbone, input_spec=[input_spec])
            
            backbone_static_path = os.path.join(save_dir, "backbone_static")
            paddle.jit.save(static_backbone, backbone_static_path)
            logger.info(f"Backbone static model saved to: {backbone_static_path}.pdmodel/.pdiparams")
        
        logger.info("=== EXPORTING HEAD TO STATIC GRAPH ===")
        if single_step_head is not None:

            logger.info("Performing deep type conversion to float32...")
            single_step_head = deep_convert_to_float32(single_step_head)

            simplified_head = SimplifiedStaticHead(single_step_head)
            simplified_head.eval()

            simplified_head = deep_convert_to_float32(simplified_head)
            
            decoder_input_spec = paddle.static.InputSpec(
                shape=[None, None], 
                dtype='int64',
                name='decoder_input_ids'
            )
            encoder_output_spec = paddle.static.InputSpec(
                shape=[None, None, single_step_head.encoder_hidden_size], 
                dtype='float32',
                name='encoder_outputs'
            )
            
            try:
                logger.info("Testing simplified head with dummy inputs...")
                dummy_decoder_ids = paddle.ones([1, 1], dtype='int64') 
                dummy_encoder_out = paddle.randn([1, 144, single_step_head.encoder_hidden_size], dtype='float32')
                
                dummy_decoder_ids = dummy_decoder_ids.cast('int64')
                dummy_encoder_out = dummy_encoder_out.cast('float32')
                
                logger.info(f"Dummy decoder ids: shape={dummy_decoder_ids.shape}, dtype={dummy_decoder_ids.dtype}")
                logger.info(f"Dummy encoder out: shape={dummy_encoder_out.shape}, dtype={dummy_encoder_out.dtype}")
                
                with paddle.no_grad():
                    test_output = simplified_head(dummy_decoder_ids, dummy_encoder_out)
                    logger.info(f"Test output shape: {test_output.shape}, dtype: {test_output.dtype}")
                    
            except Exception as e:
                logger.error(f"Test forward failed: {e}")
                logger.error("Attempting alternative approach...")
                
                try:
                    static_simplified_head = paddle.jit.to_static(
                        simplified_head, 
                        input_spec=[decoder_input_spec, encoder_output_spec]
                    )
                    
                    head_static_path = os.path.join(save_dir, "head_simplified_static")
                    paddle.jit.save(static_simplified_head, head_static_path)
                    logger.info(f"Simplified head static model saved to: {head_static_path}.pdmodel/.pdiparams")
                    
                except Exception as export_e:
                    logger.error(f"Export also failed: {export_e}")
                    head_static_path = None
            else:
                static_simplified_head = paddle.jit.to_static(
                    simplified_head, 
                    input_spec=[decoder_input_spec, encoder_output_spec]
                )
                
                head_static_path = os.path.join(save_dir, "head_simplified_static") 
                paddle.jit.save(static_simplified_head, head_static_path)
                logger.info(f"Simplified head static model saved to: {head_static_path}.pdmodel/.pdiparams")
    
    finally:
        import ppocr.modeling.heads.rec_unimernet_head as unimernet_module
        unimernet_module.AttentionMaskConverter.to_4d = originals[0]
        unimernet_module.AttentionMaskConverter._expand_mask = originals[1]
        unimernet_module.AttentionMaskConverter._make_causal_mask = originals[2]
    
    return backbone_static_path, head_static_path


class SimplifiedStaticHead(nn.Layer):
    def __init__(self, single_step_head):
        super().__init__()
        
        paddle.set_default_dtype('float32')
        
        if hasattr(single_step_head.config_decoder, '__dict__'):
            config_dict = single_step_head.config_decoder.__dict__.copy()
        else:
            config_dict = {}
            for attr in dir(single_step_head.config_decoder):
                if not attr.startswith('_'):
                    config_dict[attr] = getattr(single_step_head.config_decoder, attr)
        
        config_dict['is_export'] = True  # Force export mode
        
        from ppocr.modeling.heads.rec_unimernet_head import MBartConfig
        self.config_decoder = MBartConfig(**config_dict)
        
        self.encoder_hidden_size = single_step_head.encoder_hidden_size
        
        if hasattr(single_step_head, 'enc_to_dec_proj'):
            self.enc_to_dec_proj = single_step_head.enc_to_dec_proj
            with paddle.no_grad():
                for param in self.enc_to_dec_proj.parameters():
                    if param.dtype != paddle.float32:
                        param.set_value(param.cast(paddle.float32))
        
        from ppocr.modeling.heads.rec_unimernet_head import MBartForCausalLM
        self.decoder = MBartForCausalLM(self.config_decoder)
        
        logger.info("Transferring decoder state dict with type safety...")
        source_state_dict = single_step_head.decoder.state_dict()
        target_state_dict = {}
        
        for key, value in source_state_dict.items():
            if value.dtype != paddle.float32:
                logger.debug(f"Converting state dict {key}: {value.dtype} -> float32")
                target_state_dict[key] = value.cast(paddle.float32)
            else:
                target_state_dict[key] = value
        
        self.decoder.set_state_dict(target_state_dict)
        
        with paddle.no_grad():
            for name, param in self.decoder.named_parameters():
                if param.dtype != paddle.float32:
                    param.set_value(param.cast(paddle.float32))
                    logger.debug(f"Post-load conversion {name}: -> float32")
        
        self.eval()
        
    def forward(self, decoder_input_ids, encoder_outputs):
        batch_size = decoder_input_ids.shape[0]
        
        decoder_input_ids = decoder_input_ids.cast('int64')
        encoder_outputs = encoder_outputs.cast('float32')
        
        decoder_attention_mask = paddle.ones_like(decoder_input_ids).cast('int64')

        encoder_hidden_states = encoder_outputs
        
        if hasattr(self, 'enc_to_dec_proj') and self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.cast('float32')

        forward_kwargs = {
            'input_ids': decoder_input_ids,
            'attention_mask': decoder_attention_mask,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': None,
            'inputs_embeds': None,
            'output_attentions': False,
            'output_hidden_states': False,
            'use_cache': False,
            'past_key_values': None,
            'return_dict': True,
        }
        
        try:
            decoder_outputs = self.decoder(**forward_kwargs)
            logits = decoder_outputs.logits[:, -1, :].cast('float32')
            return logits
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Return fallback logits
            vocab_size = self.config_decoder.vocab_size
            fallback_logits = paddle.zeros([batch_size, vocab_size], dtype='float32')
            return fallback_logits



def convert_paddle_to_onnx(static_model_dir, onnx_save_path, input_shape_dict=None):
    try:
        import subprocess
        import os
        
        logger.info(f"Converting {static_model_dir} to ONNX...")
        
        model_file = static_model_dir + ".json"  
        params_file = static_model_dir + ".pdiparams"
        
        if not os.path.exists(model_file):
            logger.error(f"Model file not found: {model_file}")
            return False
        if not os.path.exists(params_file):
            logger.error(f"Params file not found: {params_file}")
            return False
        
        model_dir = os.path.dirname(static_model_dir)
        model_filename = os.path.basename(static_model_dir) + ".json" 
        params_filename = os.path.basename(static_model_dir) + ".pdiparams"
        
        cmd = [
            "paddle2onnx",
            "--model_dir", model_dir,
            "--model_filename", model_filename,
            "--params_filename", params_filename,
            "--save_file", onnx_save_path,
            "--enable_onnx_checker", "True",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted to ONNX: {onnx_save_path}")
            return True
        else:
            logger.error(f"paddle2onnx command failed: {result.stderr}")
            logger.error(f"Command output: {result.stdout}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to convert to ONNX: {e}")
        return False


def export_models_to_onnx(static_backbone_path, static_head_path, onnx_save_dir="./output/onnx_models"):
   
    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)
    
    backbone_onnx_path = None
    head_onnx_path = None
    
    if static_backbone_path and os.path.exists(static_backbone_path + ".json"):
        backbone_onnx_path = os.path.join(onnx_save_dir, "backbone.onnx")
        
        backbone_input_shape = {"image": [1, 1, 384, 384]}  
        
        success = convert_paddle_to_onnx(
            static_backbone_path, 
            backbone_onnx_path,
            backbone_input_shape
        )
        if not success:
            backbone_onnx_path = None
    
    # Convert head to ONNX  
    if static_head_path and os.path.exists(static_head_path + ".json"):
        head_onnx_path = os.path.join(onnx_save_dir, "head.onnx")
        
        # Input shape dict cho head
        head_input_shape = {
            "decoder_input_ids": [1, 1], 
            "encoder_outputs": [1, 144, 2048]  
        }
        
        success = convert_paddle_to_onnx(
            static_head_path,
            head_onnx_path, 
            head_input_shape
        )
        if not success:
            head_onnx_path = None
    
    return backbone_onnx_path, head_onnx_path


def export_and_convert_to_onnx_workflow(backbone_model, single_step_head, 
                                       static_save_dir="./output/static_models",
                                       onnx_save_dir="./output/onnx_models"):    
    logger.info("=== STARTING EXPORT AND CONVERSION WORKFLOW ===")
    
    if backbone_model is not None:
        logger.info("Deep converting backbone model...")
        validate_model_dtypes(backbone_model, "Backbone")
        backbone_model = deep_convert_to_float32(backbone_model)
        backbone_model = force_model_float32_state(backbone_model)
    
    if single_step_head is not None:
        logger.info("Deep converting single step head...")
        validate_model_dtypes(single_step_head, "SingleStepHead") 
        single_step_head = deep_convert_to_float32(single_step_head)
        single_step_head = force_model_float32_state(single_step_head)
    
    logger.info("Step 1: Exporting to static graph...")
    try:
        static_backbone_path, static_head_path = export_to_static_graph(
            backbone_model, single_step_head, static_save_dir
        )
        
        if static_head_path is None:
            logger.warning("Head export failed, continuing with backbone only...")
        
    except Exception as e:
        logger.error(f"Static graph export failed: {e}")
        return None, None
    
    backbone_onnx_path = None
    head_onnx_path = None
    
    if static_backbone_path or static_head_path:
        logger.info("Step 2: Converting to ONNX...")
        try:
            backbone_onnx_path, head_onnx_path = export_models_to_onnx(
                static_backbone_path, static_head_path, onnx_save_dir
            )
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
    
    logger.info("=== WORKFLOW COMPLETED ===")
    logger.info(f"Backbone ONNX: {backbone_onnx_path}")
    logger.info(f"Head ONNX: {head_onnx_path}")
    
    return backbone_onnx_path, head_onnx_path


def main():    
    logger.info("Building and loading original model...")
    model = build_model(config["Architecture"])
    load_model(config, model)    
    model.eval()

    logger.info("Extracting backbone and converting head to single-step mode...")
    backbone_model = model.backbone
    single_step_head = convert_head_to_single_step(model.head)
    
    export_and_convert_to_onnx_workflow(backbone_model, single_step_head)

    logger.info("ONNX export process finished.")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
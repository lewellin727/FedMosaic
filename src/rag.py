from peft import PeftModel
import torch
import gc
import copy
from src.utils import *
from src.train import MaskedLoRALinear
from peft.tuners.lora.layer import Linear as LoRALinear

def inference(question, base_model, lora_paths, tokenizer, generation_config, weights=None):
    model = base_model
    merge_cnt = len(lora_paths)
    for i in range(merge_cnt):
        lora_path = lora_paths[i]
        if i == 0:
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                adapter_name="0", 
                is_trainable=False
            )
        else:
            model.load_adapter(lora_path, adapter_name=f"{i}")
    if weights is None:
        weights = [1 / merge_cnt] * merge_cnt
    model.add_weighted_adapter(
        adapters=[str(i) for i in range(merge_cnt)], 
        weights=weights,
        adapter_name=f"merge", 
        combination_type="cat",
    )
    model.set_adapter(f"merge")

    prompt = USER_PROMPT_LORA.format(question=question, passages=None)
    with torch.no_grad():
        pred = model_generate(prompt, model, tokenizer, generation_config)

    for adapter_name in list(model.peft_config.keys()):
        model.delete_adapter(adapter_name)
    model = model.unload()
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    torch.cuda.empty_cache()
    gc.collect()

    return pred

def inference_with_mask(question, doc_scores, base_model, lora_paths, mask_paths, tokenizer, generation_config):
    
    model = base_model
    for i, (lora_path, mask_path) in enumerate(zip(lora_paths, mask_paths)):
        if i == 0:
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                adapter_name="0",
                is_trainable=False
            )
        else:
            model.load_adapter(lora_path, adapter_name=str(i))

        model.set_adapter(str(i))
        mask_dict = load_bitpacked_mask(mask_path)
        apply_mask_to_lora(model, mask_dict)
    
    merge_cnt = len(mask_paths)
    if merge_cnt > 0:
        model.add_weighted_adapter(
            adapters=[str(i) for i in range(merge_cnt)],
            weights=doc_scores,
            adapter_name="merge",
            combination_type="cat"
        )
        model.set_adapter("merge")

    prompt = USER_PROMPT_LORA.format(question=question, passages=None)
    with torch.no_grad():
        pred = model_generate(prompt, model, tokenizer, generation_config)

    model = model.unload()
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    print(type(model))
    torch.cuda.empty_cache()
    gc.collect()

    return pred


def apply_mask_to_lora(base_model, mask_dict):

    for name, module in base_model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = name.rsplit('.', 1)[0]
            attr_name = name.rsplit('.', 1)[-1]
            parent_module = dict(base_model.named_modules())[parent_name]

            mask_tensor = mask_dict.get(name)
            if mask_tensor is None:
                raise ValueError(f"Missing mask for layer {name}")
            mask_tensor = mask_tensor.to(base_model.device)
            new_module = MaskedLoRALinear(module, mask_tensor)

            setattr(parent_module, attr_name, new_module)
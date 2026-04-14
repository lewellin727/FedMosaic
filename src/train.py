from src.utils import get_prompt
from datasets import Dataset as HFDataset  
from peft import TaskType, LoraConfig, get_peft_model
from trl import SFTTrainer
import os
import gc
import copy
import torch
from transformers import TrainingArguments
import torch.nn as nn
from peft.tuners.lora.layer import Linear as LoRALinear
from peft import PeftModel
import numpy as np


class Document:
    def __init__(self, passage, aug_item, augment_model) -> None:
        self.id = aug_item['id']
        self.passage = passage
        self.rewrite = aug_item[f'{augment_model}_rewrite']
        self.augment = aug_item[f'{augment_model}_qa']
    
    def get_prompt_ids(self, tokenizer):
        prompt_ids = []
        if self.augment is None:
            return []
        qa_list_cnt = (len(self.augment) + 1) // 2
        for q_id, qa in enumerate(self.augment):
            if q_id < qa_list_cnt:
                for ppp in [self.passage, self.rewrite]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], [ppp], qa["answer"]))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], None, qa["answer"]))
        return prompt_ids


class DocCluster:
    def __init__(self, silo_id, cluster_id, Documents) -> None:
        self.silo_id = silo_id
        self.cluster_id = cluster_id
        self.Documents = Documents


    def get_train_data(self, tokenizer, max_length=1024):
        prompt_ids = []
        for doc in self.Documents:
            prompt_ids.extend(doc.get_prompt_ids(tokenizer))
        pad_token_id = tokenizer.pad_token_id or 0
        eos_token_id = tokenizer.eos_token_id
        
        dataset = []
        for input_ids in prompt_ids:
            if eos_token_id is not None and input_ids[-1] != eos_token_id:
                input_ids = input_ids + [eos_token_id]
            labels = input_ids.copy()

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            padding_len = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids += [pad_token_id] * padding_len
            labels += [-100] * padding_len

            dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            })
        
        return HFDataset.from_list(dataset) 

def train_lora(doc_cluster, base_model, tokenizer, lora_dir, config):
    silo_id = doc_cluster.silo_id
    cluster_id = doc_cluster.cluster_id
    lora_path = os.path.join(lora_dir, f'silo={silo_id}_clu={cluster_id}')

    if os.path.exists(lora_path):
        print(f'lora {silo_id}_{cluster_id} exists')
        return 

    model = base_model
    train_data = doc_cluster.get_train_data(tokenizer)
    lora_config = LoraConfig(
        r=config['train']['lora_rank'],
        lora_alpha=config['train']['lora_alpha'],
        target_modules=config['train']['target_modules'],
        lora_dropout=0,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    training_args = TrainingArguments(
        output_dir="./training_output_lora",
        save_total_limit=0, 
        per_device_train_batch_size=config['train']['batch_size'],
        gradient_accumulation_steps=1,
        learning_rate=float(config['train']['lr']),
        num_train_epochs=config['train']['lora_epoch'],
        save_strategy="no",
        # report_to="tensorboard",
        bf16=True,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        # logging_steps=10,
        # logging_strategy="steps",
        # logging_dir="./log"
    )

    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
    )
    trainer.train()

    model.save_pretrained(lora_path, safe_serialization=True)
    print(f'Saved final lora model at {lora_path}')
    
    model = model.unload()
    if isinstance(model, PeftModel):
        model = model.base_model.model
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    torch.cuda.empty_cache()
    gc.collect()


# -------------------------------- Lora with one mask ----------------------------- 

class MaskedLoRALinear(LoRALinear):
    def __init__(self, base_lora_layer: LoRALinear, mask: torch.nn.Parameter = None):
        assert len(base_lora_layer._active_adapter) == 1
        adapter_name = base_lora_layer._active_adapter[0]
        super().__init__(
            base_lora_layer.get_base_layer(),
            adapter_name=adapter_name,
            r=base_lora_layer.r[adapter_name],
            lora_alpha=base_lora_layer.lora_alpha[adapter_name],
        )

        self.lora_A = base_lora_layer.lora_A
        self.lora_B = base_lora_layer.lora_B
        self.scaling = base_lora_layer.scaling

        self.key = adapter_name
        if mask is None:
            d_in = self.lora_A[self.key].weight.shape[1]
            self.mask_A = nn.Parameter(torch.ones(1, d_in))
        else:
            self.mask_A = mask

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            # dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x_casted = x.to(lora_A.weight.dtype)

            A = lora_A.weight  # shape: (r, d_in)
            B = lora_B.weight  # shape: (d_out, r)

            mask_prob = torch.sigmoid(self.mask_A)
            mask_sample = (mask_prob > 0.5).float() + mask_prob - mask_prob.detach() 
            mask_sample = mask_sample.to(device=A.device, dtype=A.dtype)

            masked_A = A * mask_sample  # broadcast: (r, d_in) * (1, d_in)
            mask_ratio = mask_sample.sum() / mask_sample.numel()
            masked_A = masked_A / mask_ratio 

            x_proj = x_casted @ masked_A.T           # (B, *, r)
            lora_out = x_proj @ B.T * scaling        # (B, *, d_out)

            result = result + lora_out

        return result.to(torch_result_dtype)
    


class MaskedLoraTrainer(SFTTrainer):
    def __init__(self, *args, lambda_l1=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_l1 = lambda_l1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        
        l1_loss = 0.0
        for name, param in model.named_parameters():
            if 'mask_A' in name:
                l1_loss += torch.norm(torch.sigmoid(param * 10), p=1)
        loss += self.lambda_l1 * l1_loss

        return (loss, outputs) if return_outputs else loss


def train_mask_with_fixed_lora(silo_id, clu_id, doc, base_model, tokenizer, init_lora_path, save_mask_dir, config):
    save_mask_path = os.path.join(save_mask_dir, f'doc_id={doc.id}.pt')
    # if os.path.exists(save_mask_path):
    #     return
    one_doc_cluster = DocCluster(silo_id, clu_id, [doc])
    train_data = one_doc_cluster.get_train_data(tokenizer)

    base_model_copy = base_model

    model = PeftModel.from_pretrained(base_model_copy, init_lora_path, is_trainable=True)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = name.rsplit('.', 1)[0]
            attr_name = name.rsplit('.', 1)[-1]
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, attr_name, MaskedLoRALinear(module))

    for name, param in model.named_parameters():
        if any(t in name for t in ['mask_A']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print(f'Traingable params: {trainable_params}')

    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:      
            total_trainable += param.numel()
    print(f"Total trainable parameters: {total_trainable:,}")
    
    
    training_args = TrainingArguments(
        output_dir=f"./training_output_masked",
        save_total_limit=0, 
        per_device_train_batch_size=config['train']['mask_batch_size'],
        gradient_accumulation_steps=1,
        learning_rate=float(config['train']['mask_lr']),
        num_train_epochs=config['train']['mask_epoch'],
        save_strategy="no",
        report_to="none",
        bf16=True,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        # logging_steps=1,
        # logging_strategy="steps",
        # logging_dir="./log",
        # report_to="tensorboard"
    )

    trainer = MaskedLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        lambda_l1=float(config['train'].get('lambda_l1', 1e-4)),
    )

    trainer.train()


    bitpacked_mask_dict = {}
    one_cnt = 0
    total_cnt = 0

    for name, module in model.named_modules():
        if isinstance(module, MaskedLoRALinear):
            mask = (torch.sigmoid(module.mask_A.detach().cpu()) > 0.5).numpy().astype(np.uint8)
            one_cnt += np.count_nonzero(mask)
            total_cnt += mask.size

            flat = mask.flatten()
            bitpacked = np.packbits(flat, bitorder='little') 
            bitpacked_mask_dict[name] = {
                'bitpack': bitpacked,
                'shape': mask.shape,
                'length': flat.size  
            }

    save_mask_path = os.path.join(save_mask_dir, f'doc_id={doc.id}.pt')
    os.makedirs(save_mask_dir, exist_ok=True)
    torch.save(bitpacked_mask_dict, save_mask_path)
    print(f'Mask Ratio = {one_cnt / total_cnt:.8f}')
    print(f'Mask Saved to {save_mask_path}')

    if isinstance(model, PeftModel):
        model = model.base_model.model
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    print(type(model))
    torch.cuda.empty_cache()
    gc.collect()


def train_mask_with_fixed_lora_cluster(doc_cluster, model, tokenizer, lora_save_dir, save_mask_dir, config):
    for doc in doc_cluster.Documents:
        silo_id, cluster_id = doc_cluster.silo_id, doc_cluster.cluster_id
        init_lora_path = os.path.join(lora_save_dir, f'silo={silo_id}_clu={cluster_id}')
        save_mask_path = os.path.join(save_mask_dir, f'doc_id={doc.id}.pt')
        train_mask_with_fixed_lora(silo_id, cluster_id, doc, model, tokenizer, init_lora_path, save_mask_dir, config)



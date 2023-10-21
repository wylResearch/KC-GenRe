# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import string
import pdb

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    Trainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft.utils.config import PromptLearningConfig, PeftType
import pdb

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
PAD_SCORE_VALUE = 100000 
OPTIONS_LETTER = string.ascii_uppercase + string.ascii_lowercase



OPTION_LETTER_IDS_TOP10 = [29909, 29933, 29907, 29928, 29923, 29943, 29954, 29950, 29902, 29967]  # ['A', 'B', ...,'J']
OPTION_LETTER_IDS_TOP20 = [29909, 29933, 29907, 29928, 29923, 29943, 29954, 29950, 29902, 29967, 
                           29968, 29931, 29924, 29940, 29949, 29925, 29984, 29934, 29903, 29911]
OPTION_GEN_TOKEN_IDS = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="huggyllama/llama-7b" 
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(                    # 
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset_train: str = field(
        default='./dataset/ReVerb20K_rerank_data_train_v2.json',           # KG dataset path
        metadata={"help": "Path of train dataset to finetune on."}
    )
    dataset_eval: str = field(
        default='./dataset/ReVerb20K_rerank_data_valid_v2.json',           # KG dataset path
        metadata={"help": "Path of valid dataset."}
    )
    dataset_test: str = field(
        default='./dataset/ReVerb20K_rerank_data_test_v2.json',            # KG dataset path
        metadata={"help": "Path of test dataset."}
    )
    
    # save file name
    save_file_name: Optional[str] = field(default="")
    use_option_scores: Optional[bool] = field(default=False, metadata={"help": 'use option_scores to compute rank loss'})
    use_constraint_gen: Optional[bool] = field(default=False, metadata={"help": 'use constraint generation'})
    gen_TopK: Optional[int] = field(default=10)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

    rankloss_weight: float = field(default=0., metadata={"help": 'weight of rank loss'}) #
    add_option_loss:  bool = field(default=False, metadata={"help": ''})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 



def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint:", checkpoint_dir)
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    use_constraint_gen: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        if not self.predict_with_generate:
            targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
            tokenized_targets = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
        # Build the input and labels for causal LM
        input_ids = [] 
        option_scores_flag = True if 'option_scores' in instances[0] else False
        if not self.predict_with_generate:      # train/eval
            option_scores = []
            labels = []
            for i, (tokenized_source, tokenized_target) in enumerate(zip(
                tokenized_sources_with_prompt['input_ids'], 
                tokenized_targets['input_ids']
            )):
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
                if option_scores_flag:
                    option_scores.append(torch.tensor(instances[i]['option_scores']))
        else:       # test
            labels = None
            for tokenized_source in tokenized_sources_with_prompt['input_ids']:
                input_ids.append(torch.tensor(tokenized_source))
        
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:  # train/eval
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
            data_dict['labels'] = labels
        if option_scores_flag:
            option_scores = pad_sequence(option_scores, batch_first=True, padding_value=PAD_SCORE_VALUE) # padding_value 对应没有选项的分数, 设成 100000（第一阶段的分数越大排序越靠后）
            options = [letter for letter in OPTIONS_LETTER[:option_scores.shape[1]]]
            option_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(options))
            data_dict['option_scores'] = option_scores
            data_dict['option_token_id'] = option_token_id
        
        if self.use_constraint_gen:
            data_dict['prefix_allowed_tokens_fn'] = my_prefix_allowed_tokens_fn

        return data_dict




def local_dataset(dataset_name):
    if dataset_name.endswith('.json'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(filename=dataset_name, format='jsonlines')
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")
    return full_dataset 

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Load local datasets.
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """
    def load_data(dataset_name):
        if os.path.exists(dataset_name):
            try:
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset is not exist: {dataset_name}.")

    def format_dataset(dataset, use_option_scores):
        if use_option_scores:
            use_columns = ['input', 'output', 'option_scores']
        else:
            use_columns = ['input', 'output']

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in use_columns]     # 很多列
        )
        return dataset
        
     # Load dataset.
    if args.do_train:
        train_dataset = load_data(args.dataset_train)
        train_dataset = format_dataset(train_dataset, args.use_option_scores)    
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_eval:   
        eval_dataset = load_data(args.dataset_eval)
        eval_dataset = format_dataset(eval_dataset, args.use_option_scores)     
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_predict:
        test_dataset = load_data(args.dataset_test)
        test_dataset = format_dataset(test_dataset, args.use_option_scores)      
        if args.max_test_samples is not None and len(test_dataset) > args.max_test_samples:
            test_dataset = test_dataset.select(range(args.max_test_samples))
        if args.group_by_length:
            test_dataset = test_dataset.map(lambda x: {'length': len(x['input'])})
    
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer, 
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
        use_constraint_gen=args.use_constraint_gen,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None, 
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=test_dataset if args.do_predict else None,      # 
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # if is_completed: return None, True # already finished       # TODO
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training





class RanklossTrainer(Seq2SeqTrainer):
    def gather_logits_labels(self, logits, option_token_id, labels):
        ########## 
        #  '\n' position
        token_id = self.tokenizer.convert_tokens_to_ids(["<0x0A>"])[0]      # TODO
        indices = torch.argmax(torch.eq(labels, token_id).float(), dim=1)     # TODO important
        # option position
        indices += 1  

        rows  = torch.arange(labels.shape[0])
        logits_ = logits[rows, indices, :]
        logits_option_token = torch.gather(logits_, 1, option_token_id.unsqueeze(0).repeat(logits_.shape[0], 1))
        return logits_option_token



    def rank_loss(self, pred_scores, label_scores, label_scores_raw):
        rank_loss = torch.zeros(1).type(pred_scores.dtype).to(pred_scores.device)
        for i in range(pred_scores.shape[0]):
            x = pred_scores[i]
            y = label_scores[i].reshape(1,-1)
            diff = x.unsqueeze(0) - x.unsqueeze(-1) # b * b
            rw_diff = y.unsqueeze(0) - y.unsqueeze(-1) # b * b
            aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
            rank_loss_i = -diff[aval].sum()
            rank_loss += rank_loss_i
            
            if self.args.add_option_loss:
                mask = label_scores_raw[i] == PAD_SCORE_VALUE
                rank_loss_i_option = pred_scores[i][mask].sum()
                rank_loss += rank_loss_i_option

        rank_loss = rank_loss/pred_scores.shape[0]
        

        k = aval.shape[0]   # rerank_topk
        rank_loss = rank_loss*100/(k*k)     # scale
        return rank_loss


    
    def rescale_scores_min_max(self, option_scores, reverse=False, replace_pad=False):
        if replace_pad:
            replace_mask = option_scores == PAD_SCORE_VALUE
            option_scores_masked = option_scores.masked_fill(replace_mask, -float("inf"))
            max_values, _ = torch.max(option_scores_masked, dim=1)
            max_values = max_values.unsqueeze(1).repeat(1, option_scores.shape[1])
            option_scores_ = torch.where(replace_mask, max_values, option_scores)
        else:
            option_scores_ = option_scores

        option_scores_min = option_scores_.min(dim=1, keepdim=True).values
        option_scores_max = option_scores_.max(dim=1, keepdim=True).values
        option_scores_normalized = (option_scores_ - option_scores_min) / (option_scores_max - option_scores_min)   
        
        if reverse:
            option_scores_normalized = 1 - option_scores_normalized
        
        nan_mask = torch.isnan(option_scores_normalized)
        option_scores_normalized = torch.where(nan_mask, torch.ones_like(option_scores_normalized), option_scores_normalized)
        return option_scores_normalized


    def compute_loss(self, model, inputs, return_outputs=False):
        ##############  compute_loss (raw) ##############
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # outputs = model(**inputs) 
        outputs = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),
                        labels=inputs.get("labels"))   # TODO changed

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        
        ##############  compute_loss (rank) ##############
        if "option_scores" in inputs:
            option_scores = inputs.get('option_scores')
            logits = outputs.logits
            pred_scores = self.gather_logits_labels(logits, inputs.get("option_token_id"), inputs.get("labels"))
            pred_scores_ = self.rescale_scores_min_max(pred_scores)
            option_scores_ = self.rescale_scores_min_max(option_scores, reverse=self.args.reverse_option_scores, replace_pad=True)
            rank_loss = torch.squeeze(self.rank_loss(pred_scores_, option_scores_, option_scores))
            loss = self.args.rankloss_weight * rank_loss + loss

        return (loss, outputs) if return_outputs else loss



def my_prefix_allowed_tokens_fn(batch_idx, input_ids):
    """
    self.tokenizer.convert_ids_to_tokens(
    [22125, 292, 29901, 
    13, 29909, 29889, 29871, A
    13, 29933, 29889, 29871, B
    13, 29907, 29889, 29871, C
    13, 29928, 29889, 29871, D
    13, 29923, 29889, 29871, E
    13, 29943, 29889, 29871, F
    13, 29954, 29889, 29871, G
    13, 29950, 29889, 29871, H
    13, 29902, 29889, 29871, I
    13, 29967, 29889, 2])    J
    ['▁Rank', 'ing', ':', 
    '<0x0A>', 'A', '.', '▁',  
    '<0x0A>', 'B', '.', '▁', 
    '<0x0A>', 'C', '.', '▁', 
    '<0x0A>', 'D', '.', '▁', 
    '<0x0A>', 'E', '.', '▁', 
    '<0x0A>', 'F', '.', '▁', 
    '<0x0A>', 'G', '.', '▁', 
    '<0x0A>', 'H', '.', '▁', 
    '<0x0A>', 'I', '.', '▁', 
    '<0x0A>', 'J', '.', '</s>']
    """
    # 


    PAD_ID = 32000      
    INPUT_END_IDS = [13291, 29901]     
    ANSWER_PREFIX_ID = [22125, 292, 29901]
    OPTION_LEN = 4
    OPTION_SPECIAL_IDS = [13, 29889, 29871, 2]      # '<0x0A>', '.', '▁', '</s>'    


    indices = torch.nonzero(input_ids == INPUT_END_IDS[0])
    indices = indices.tolist()
    input_ids_ = input_ids.tolist()
    start_idx = 0
    for idx in indices:
        idx = idx[0]
        if input_ids_[idx:(idx+2)] == INPUT_END_IDS:
            start_idx = idx
    
    start_idx += len(INPUT_END_IDS)
    ans_ids = input_ids_[start_idx:]

    ans_ids = [ans_id for ans_id in ans_ids if ans_id != PAD_ID]

    allowed_tokens = []

    allowed_tokens_prefix_dict = {}
    for i,id in enumerate(ANSWER_PREFIX_ID):
        allowed_tokens_prefix_dict[i] = [id]

    ####### TODO 
    ans_ids_len = len(ans_ids)
    prefix_len = len(ANSWER_PREFIX_ID)
    total_len = prefix_len + OPTION_LEN * len(OPTION_GEN_TOKEN_IDS)
    if ans_ids_len < prefix_len:
        allowed_tokens = allowed_tokens_prefix_dict[ans_ids_len]
    elif ans_ids_len >= prefix_len and (ans_ids_len-prefix_len)%OPTION_LEN==0: # '<0x0A>'
        allowed_tokens = OPTION_SPECIAL_IDS[0]
    elif ans_ids_len >= prefix_len and (ans_ids_len-prefix_len)%OPTION_LEN==1: # letters
        allowed_tokens = list(set(OPTION_GEN_TOKEN_IDS) - set(ans_ids))
    elif ans_ids_len >= prefix_len and (ans_ids_len-prefix_len)%OPTION_LEN==2: # '.'
        allowed_tokens = OPTION_SPECIAL_IDS[1]
    elif ans_ids_len >= prefix_len and (ans_ids_len-prefix_len)%OPTION_LEN==3:
        if ans_ids_len < total_len-1: # '▁'
            allowed_tokens = OPTION_SPECIAL_IDS[2]
        elif ans_ids_len == total_len-1: # '</s>'
            allowed_tokens = OPTION_SPECIAL_IDS[3]
    
    if ans_ids_len >= total_len:
        return []

    # pdb.set_trace()
    return allowed_tokens


    

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if 'FB15K-237-N' in args.dataset_train or 'Wiki27K' in args.dataset_train:
        training_args.reverse_option_scores = False     # 
    elif 'ReVerb20K' in args.dataset_train or 'ReVerb45K' in args.dataset_train:
        training_args.reverse_option_scores = True      # 

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary. 
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(                    
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })


    if args.use_constraint_gen:
        # generate -> peft.peft_model -> PeftModelForCausalLM  generate 
        # -> transformers.generation.utils generate
        options_gen = [letter for letter in OPTIONS_LETTER[:args.gen_TopK]]
        # pdb.set_trace()
        global OPTION_GEN_TOKEN_IDS
        OPTION_GEN_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(options_gen))

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    # trainer = Seq2SeqTrainer(
    #     model=model, 
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    # )
    trainer = RanklossTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        )


    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not. 
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'rerank_predictions_'+args.save_file_name+'.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "rerank_predict_metrics_"+args.save_file_name+".json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()

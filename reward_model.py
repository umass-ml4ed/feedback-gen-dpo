from typing import List
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5EncoderModel, T5PreTrainedModel, T5Model, PreTrainedModel, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from reward_model_dataset import LABELS
from utils import device

peft_config = LoraConfig(
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="SEQ_CLS",
    inference_mode=False,
)

bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype="float16",
    # bnb_4bit_use_double_quant=False,
    llm_int8_skip_modules=["score"],
    load_in_8bit=True,
)

class T5Classifier(T5PreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(p=config.classifier_dropout),
            nn.Linear(config.d_model, config.num_labels)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()
        self.model_parallel = False

    def forward(self, input_ids, attention_mask, labels = None, **kwargs):
        model_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        seq_lens = attention_mask.sum(dim=1)
        masked_hidden_states = model_outputs.last_hidden_state * attention_mask.unsqueeze(2)
        avg_hidden_states = masked_hidden_states.sum(dim=1) / seq_lens.unsqueeze(1)
        logits = self.classifier(avg_hidden_states)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class T5ClassifierWDecoder(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = T5Model(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(p=config.classifier_dropout),
            nn.Linear(config.d_model, config.num_labels)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()
        self.model_parallel = False

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = None, **kwargs):
        decoder_input_ids = nn.functional.pad(decoder_input_ids, (1, 0), value=self.config.decoder_start_token_id)
        decoder_attention_mask = nn.functional.pad(decoder_attention_mask, (1, 0), value=1)
        model_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        seq_lens = decoder_attention_mask.sum(dim=1)
        hidden_states = model_output.last_hidden_state[torch.arange(input_ids.shape[0]), seq_lens - 1]
        logits = self.classifier(hidden_states)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels = None, **kwargs):
        all_logits = [
            model(input_ids, attention_mask, labels=labels).logits
            for model in self.models
        ]
        logits = torch.stack(all_logits).mean(dim=0)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

def get_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "llama" in base_model or "meta-math" in base_model:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer

def get_reward_model(model_name: str, base_model: str, tokenizer, enc_dec: bool, test: bool):
    if "t5" in base_model:
        if enc_dec:
            model = T5ClassifierWDecoder.from_pretrained(model_name, num_labels=len(LABELS), classifier_dropout=0.0).to(device)
        else:
            model = T5Classifier.from_pretrained(model_name, num_labels=len(LABELS), classifier_dropout=0.0).to(device)
    elif "llama" in base_model or "meta-math" in base_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(LABELS),
            problem_type="multi_label_classification",
            pad_token_id=tokenizer.pad_token_id,
            quantization_config=None if test else bnb_config,
            # Higher precision for non-quantized parameters helps training accuracy and doesn't hurt performance
            # Lower precision at test time improves speed and only marginally hurts performance
            torch_dtype=torch.float16 if test else torch.float32,
            device_map={"": 0}
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        if test:
            model = PeftModel.from_pretrained(model, model_name).merge_and_unload()
        else:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(LABELS), problem_type="multi_label_classification").to(device)
    return model

def get_ensemble(model_names: List[str], base_model: str, tokenizer, enc_dec: bool, test: bool):
    models = [
        get_reward_model(model_name, base_model, tokenizer, enc_dec, test)
        for model_name in model_names
    ]
    return EnsembleModel(models)

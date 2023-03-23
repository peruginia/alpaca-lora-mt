import os

import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers

from dataclasses import dataclass, field
from typing import Optional
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

PROMPTS = {
    "en": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    },
    "pt": {
        "prompt_input": (
            "Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. "
            "Escreva uma resposta que complete adequadamente o pedido.\n\n"
            "### Instrução:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:\n"
        ),
        "prompt_no_input": (
            "Abaixo está uma instrução que descreve uma tarefa. "
            "Escreva uma resposta que complete adequadamente o pedido.\n\n"
            "### Instrução:\n{instruction}\n\n### Resposta:\n"
        ),
    },
    "es": {
        "prompt_input": (
            "A continuación se muestra una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. "
            "Escribe una respuesta que complete adecuadamente la petición.\n\n"
            "### Instrucción:\n{instruction}\n\n### Entrada:\n{input}\n\n### Respuesta:\n"
        ),
        "prompt_no_input": (
            "A continuación se muestra una instrucción que describe una tarea. "
            "Escribe una respuesta que complete adecuadamente la petición.\n\n"
            "### Instrucción:\n{instruction}\n\n### Respuesta:\n"
        ),
    },
    "ca": {
        "prompt_input": (
            "A continuació es mostra una instrucció que descriu una tasca, juntament amb una entrada que proporciona més context. "
            "Escriviu una resposta que completi adequadament la petició.\n\n"
            "### Instrucció:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:\n"
        ),
        "prompt_no_input": (
            "A continuació es mostra una instrucció que descriu una tasca. "
            "Escriviu una resposta que completi adequadament la petició.\n\n"
            "### Instrucció:\n{instruction}\n\n### Resposta:\n"
        ),
    },
    "eu": {
        "prompt_input": (
            "Azpian ataza bat deskribatzen duen instruzio bat dago, testuinguru gehiago ematen duen sarrera batekin batera. "
            "Idatzi eskaera behar bezala betetzen duen erantzuna.\n\n"
            "### Instrukzioa:\n{instruction}\n\n### Sarrera:\n{input}\n\n### Erantzuna:\n"
        ),
        "prompt_no_input": (
            "Azpian ataza bat deskribatzen duen instruzio bat dago. "
            "Idatzi eskaera behar bezala betetzen duen erantzuna.\n\n"
            "### Instrukzioa:\n{instruction}\n\n### Erantzuna:\n"
        ),
    },
    "gl": {
        "prompt_input": (
            "A seguinte é unha instrución que describe unha tarefa, xunto cunha entrada que proporciona máis contexto. "
            "Escriba unha resposta que complete correctamente a solicitude.\n\n"
            "### Instrución:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:\n"
        ),
        "prompt_no_input": (
            "A seguinte é unha instrución que describe unha tarefa. "
            "Escriba unha resposta que complete correctamente a solicitude.\n\n"
            "### Instrución:\n{instruction}\n\n### Resposta:\n"
        ),
    },
    "at": {
        "prompt_input": (
            "De siguío amuésase una instrucción que describe una xera, xuntu con una entrada qu'apurre más contestu. "
            "Escribe una respuesta que complete afechiscamente'l pidimientu.\n\n"
            "### Instrucción:\n{instruction}\n\n## Entrada:\n{input}\n\n### Respuesta:\n"
        ),
        "prompt_no_input": (
            "De siguío amuésase una instrucción que describe una xera. "
            "Escribe una respuesta que complete afechiscamente'l pidimientu.\n\n"
            "### Instrucción:\n{instruction}\n\n## Respuesta:\n"
        ),
    },
}


def load_model_tokenizer(model_args):
    """Load the model and tokenizer from the model name or path.

    Args:
        model_args: The model arguments.
    Returns:
        model: The model.
        tokenizer: The tokenizer.
    """
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
        )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    return model, tokenizer


def generate_prompt(data_point, lang):
    lang = "en"
    if data_point["input"]:
        return PROMPTS[lang]["prompt_input"].format_map(data_point)
    else:
        return PROMPTS[lang]["prompt_no_input"].format_map(data_point)


def generate_and_tokenize_prompt(data_point, lang, tokenizer, block_size):
    user_prompt = generate_prompt(data_point, lang) + data_point["output"]
    user_prompt = tokenizer(
        user_prompt, truncation=True, max_length=block_size, return_tensors=None
    )
    if (
        user_prompt["input_ids"][-1] != tokenizer.eos_token_id
        and len(user_prompt["input_ids"]) < block_size
    ):
        user_prompt["input_ids"].append(tokenizer.eos_token_id)
        user_prompt["attention_mask"].append(1)

    user_prompt["labels"] = user_prompt["input_ids"].copy()

    return user_prompt


def load_data(data_args, tokenizer):
    data = DatasetDict()
    for lang in data_args.dataset_config_names:
        data[lang] = load_dataset(data_args.dataset_name, lang)

    datasets = []
    for lang in data:
        train_data = (
            data[lang]["train"]
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(
                    x, lang, tokenizer, data_args.block_size
                )
            )
        )
        datasets.append(train_data)

    dataset = concatenate_datasets(datasets)

    train_val = dataset.train_test_split(
        test_size=data_args.validation_split_percentage / 100, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]

    return train_data, val_data


def train(model_args, data_args, training_args, model, tokenizer, train_data, val_data):
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    model.config.to_json_file(
        os.path.join(training_args.output_dir, "adapter_config.json")
    )

    trainer.train()

    trainer.save_model()

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_names is not None:
            kwargs["dataset_args"] = data_args.dataset_config_names
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_names}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    model.save_pretrained(
        training_args.output_dir,
        push_to_hub=training_args.push_to_hub,
        repo_id=training_args.hub_model_id,
    )

    print("\n If there's a warning about missing keys above, please disregard :)")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    lora_r: Optional[int] = field(
        default=8,
    )
    lora_alpha: Optional[float] = field(
        default=16,
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
    )
    target_modules: Optional[str] = field(
        default_factory=list,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_names: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_files: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_tokenizer(model_args)
    train_files, validation_file = load_data(data_args, tokenizer)
    train(
        model_args,
        data_args,
        training_args,
        model,
        tokenizer,
        train_files,
        validation_file,
    )


if __name__ == "__main__":
    main()

import torch
from peft import PeftModel
import transformers
import gradio as gr

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation import top_k_top_p_filtering

LANG_CODES = {
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "Catalan": "ca",
    "Basque": "eu",
    "Galician": "gl",
    "Asturian": "at",
}

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

RESPONSE = {
    "en": "Response",
    "pt": "Resposta",
    "es": "Respuesta",
    "ca": "Resposta",
    "eu": "Erantzuna",
    "gl": "Resposta",
    "at": "Respuesta",
}

EXAMPLES = {
    "en": [
        ["Tell me about alpacas.", "English"],
        ["Tell me about the president of Mexico in 2019.", "English"],
        ["Tell me about the king of France in 2019.", "English"],
        ["List all Canadian provinces in alphabetical order.", "English"],
        [
            "Write a Python program that prints the first 10 Fibonacci numbers.",
            "English",
        ],
        [
            "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
            "English",
        ],
        ["Tell me five words that rhyme with 'shock'.", "English"],
        [
            "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
            "English",
        ],
    ],
    "pt": [
        ["Conte-me sobre alpacas.", "Portuguese"],
        ["Conte-me sobre o presidente do México em 2019.", "Portuguese"],
        ["Conte-me sobre o rei da França em 2019.", "Portuguese"],
        [
            "Liste todas as províncias canadenses em ordem alfabética.",
            "Portuguese",
        ],
        [
            "Escreva um programa Python que imprima os primeiros 10 números de Fibonacci.",
            "Portuguese",
        ],
        [
            "Escreva um programa que imprima os números de 1 a 100. Mas para múltiplos de três imprima 'Fizz' em vez do número e para múltiplos de cinco imprima 'Buzz'. Para números que são múltiplos de três e cinco imprima 'FizzBuzz'.",
            "Portuguese",
        ],
        ["Diga-me cinco palavras que rimam com 'choque'.", "Portuguese"],
        [
            "Traduza a frase 'Não tenho boca, mas devo gritar' para o espanhol.",
            "Portuguese",
        ],
    ],
    "es": [
        ["Háblame de las alpacas.", "Spanish"],
        ["Háblame del presidente de México en 2019.", "Spanish"],
        ["Háblame del rey de Francia en 2019.", "Spanish"],
        [
            "Lista de todas las provincias canadienses en orden alfabético.",
            "Spanish",
        ],
        [
            "Escribe un programa en Python que imprima los primeros 10 números de Fibonacci.",
            "Spanish",
        ],
        [
            "Escriba un programa que imprima los números del 1 al 100. Pero para los múltiplos de tres, escriba 'Fizz' en lugar del número y para los múltiplos de cinco, escriba 'Buzz'. Para los números que son múltiplos de tres y de cinco, escriba 'FizzBuzz'.",
            "Spanish",
        ],
        ["Dime cinco palabras que rimen con 'choque'.", "Spanish"],
        [
            "Traducir la oración 'No tengo boca pero debo gritar' al inglés.",
            "Spanish",
        ],
    ],
    "ca": [
        ["Parla'm de les alpaques.", "Catalan"],
        ["Parla'm del president de Mèxic el 2019.", "Catalan"],
        ["Parla'm del rei de França el 2019.", "Catalan"],
        [
            "Llista totes les províncies canadenques per ordre alfabètic.",
            "Catalan",
        ],
        [
            "Escriu un programa Python que imprimeixi els primers 10 nombres de Fibonacci.",
            "Catalan",
        ],
        [
            "Escriu un programa que imprimeixi els números de l'1 al 100. Però per a múltiples de tres escriviu 'Fizz' en comptes del nombre i per als múltiples de cinc imprimiu 'Buzz'. Per als números que són múltiples de tres i cinc imprimiu 'FizzBuzz'.",
            "Catalan",
        ],
        ["Digues-me cinc paraules que rimin amb 'xoc'.", "Catalan"],
        [
            "Tradueix la frase 'No tinc boca, però he de cridar' al castellà.",
            "Catalan",
        ],
    ],
    "eu": [
        ["Hitz egin alpakei buruz.", "Basque"],
        ["Hitz egin 2019ko Mexikoko presidenteaz.", "Basque"],
        ["Hitz egin 2019ko Frantziako erregeaz.", "Basque"],
        ["Zerrendatu Kanadako probintzia guztiak ordena alfabetikoan.", "Basque"],
        [
            "Idatzi lehen 10 Fibonacci zenbakiak inprimatzen dituen Python programa bat.",
            "Basque",
        ],
        [
            "Idatzi 1etik 100era bitarteko zenbakiak inprimatzen dituen programa bat. Baina hiruren multiploetarako inprimatu 'Fizz' zenbakiaren ordez eta bosten multiploetarako inprimatu 'Buzz'. Hiru eta bosten multiploak diren zenbakietarako inprimatu 'FizzBuzz'.",
            "Basque",
        ],
        ["Esan 'deskarga' hitzaren errima duten bost hitz.", "Basque"],
        [
            "Itzuli 'Ez dut ahorik baina garrasi egin behar dut' esaldia gaztelaniara.",
            "Basque",
        ],
    ],
    "gl": [
        ["Fálame de alpacas.", "Galician"],
        ["Fálame do presidente de México en 2019.", "Galician"],
        ["Fálame do rei de Francia en 2019.", "Galician"],
        [
            "Enumere todas as provincias canadenses por orde alfabética.",
            "Galician",
        ],
        [
            "Escribe un programa Python que imprima os primeiros 10 números de Fibonacci.",
            "Galician",
        ],
        [
            "Escribe un programa que imprima os números do 1 ao 100. Pero para múltiplos de tres escribe 'Fizz' en lugar do número e para os múltiplos de cinco imprime 'Buzz'. Para números que son múltiplos de tres e cinco imprime 'FizzBuzz'.",
            "Galician",
        ],
        ["Dime cinco palabras que riman con 'choque'.", "Galician"],
        [
            "Traducir ao castelán a frase 'Non teño boca pero debo berrar'.",
            "Galician",
        ],
    ],
    "at": [
        ["Fálame de les alpaques.", "Asturian"],
        ["Fálame del presidente de México en 2019.", "Asturian"],
        ["Fálame del rei de Francia en 2019.", "Asturian"],
        ["Llista de toles provincies canadienses n'orde alfabéticu.", "Asturian"],
        [
            "Escribe un programa en Python qu'imprima los primeres 10 númberos de Fibonacci.",
            "Asturian",
        ],
        [
            "Escriba un programa qu'imprima los númberos del 1 al 100. Pero para los múltiplos de trés, escriba 'Fizz' en llugar del númberu y para los múltiplos de cinco, escriba 'Buzz'. Para los númberos que son múltiplos de trés y de cinco, escriba 'FizzBuzz'.",
            "Asturian",
        ],
        ["Dime cinco palabres que rimen con 'choque'.", "Asturian"],
        [
            "Traducir la oración 'Nun tengo boca pero tengo de glayar' al español.",
            "Asturian",
        ],
    ],
}

BASE_MODELS = [
    "decapoda-research/llama-7b-hf",
    # "decapoda-research/llama-13b-hf",
    # "decapoda-research/llama-30b-hf",
    # "decapoda-research/llama-65b-hf",
]

BASE_LORA_MODELS = {
    "decapoda-research/llama-7b-hf": {
        "Multilingual": "HiTZ/alpaca-lora-7b-en-pt-es-ca-eu-gl-at",
        "English": "HiTZ/alpaca-lora-7b-en",
        "Portuguese": "HiTZ/alpaca-lora-7b-pt",
        "Spanish": "HiTZ/alpaca-lora-7b-es",
        "Catalan": "HiTZ/alpaca-lora-7b-ca",
        "Basque": "HiTZ/alpaca-lora-7b-eu",
        "Galician": "HiTZ/alpaca-lora-7b-gl",
        "Asturian": "HiTZ/alpaca-lora-7b-at",
    }
}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def load_model_tokenizer(base_model_name, lora_model_name):
    tokenizer = LlamaTokenizer.from_pretrained(base_model_name)

    if device == "cuda":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_model = PeftModel.from_pretrained(
            base_model, lora_model_name, torch_dtype=torch.float16
        )
    elif device == "mps":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_name,
            device_map={"": device},
        )

    base_model.eval()
    lora_model.eval()
    """
    if torch.__version__ >= "2":
        model = torch.compile(model)
    """
    return base_model, lora_model, tokenizer


def load_base_model(base_model_name):
    if device == "cuda":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )

    base_model.eval()
    """
    if torch.__version__ >= "2":
        model = torch.compile(model)
    """
    return base_model


def load_lora_model(base_model, lora_model_name):
    if device == "cuda":
        lora_model = PeftModel.from_pretrained(
            base_model, lora_model_name, torch_dtype=torch.float16
        )
    elif device == "mps":
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_name,
            device_map={"": device},
        )

    lora_model.eval()
    """
    if torch.__version__ >= "2":
        model = torch.compile(model)
    """
    return lora_model


tokenizer = LlamaTokenizer.from_pretrained(BASE_MODELS[0])

base_models = {}
for base_model_name in BASE_MODELS:
    print("Loading base model: " + base_model_name)
    base_models[base_model_name] = load_base_model(base_model_name)


def generate_prompt(data_point, lang):
    lang = "en"
    if data_point["input"]:
        return PROMPTS[lang]["prompt_input"].format_map(data_point)
    else:
        return PROMPTS[lang]["prompt_no_input"].format_map(data_point)


def run_model(model, input_ids: torch.tensor, decoder_args):
    gen_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids, **decoder_args
    )
    decoder_outputs = model(
        **gen_inputs,
    )

    # Get the logits for the possible words
    decoder_args = model._update_model_kwargs_for_generation(
        decoder_outputs,
        decoder_args,
        is_encoder_decoder=False,
    )

    return decoder_outputs.logits, decoder_args


def test_GPT_unconstrained(
    tokenizer,
    model,
    sentence,
    language,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 40,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    generation_mode: str = "greedy",
):

    # sentence = " ".join(sentence.strip().split())

    input_text = sentence

    with torch.no_grad():
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        encoder_output = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )

        decoder_args = {
            "attention_mask": model_inputs["attention_mask"],
            "use_cache": True,
            "encoder_outputs": encoder_output,
        }

        inputs_ids = model_inputs["input_ids"]

        while (
            len(inputs_ids[0]) < max_new_tokens
            and inputs_ids[0][-1] != tokenizer.eos_token_id
        ):
            # print(inputs_ids)
            logits, decoder_args = run_model(
                model,
                inputs_ids,
                decoder_args,
            )

            if generation_mode == "greedy":
                logits = logits[:, -1, :]
                logits = torch.nn.functional.softmax(logits, dim=-1)
                next_word_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            elif generation_mode == "multinomial":
                logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(
                    logits,
                    top_k=top_k,
                    top_p=top_p,
                )
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_word_id = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(
                    f'generation_mode "{generation_mode}" not supported. Use greedy or multinomial.'
                )

            inputs_ids = torch.cat([inputs_ids, next_word_id], dim=1)
            # print([15948, 27, 27, 27], logits[0, [15948, 27, 27, 27]])
            # print(tokenizer.decode(next_word_id[0]), logits[0, next_word_id])
            output = tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)[0]
            yield output.split(f"### {RESPONSE[LANG_CODES[language]]}:")[1].strip()


def evaluate(
    instruction,
    language,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_length=256,
    num_beams=4,
    decoding_strategy="multinomial",
    **kwargs,
):
    model = load_lora_model(
        base_models[BASE_MODELS[0]], BASE_LORA_MODELS[BASE_MODELS[0]][language]
    )
    data_point = {"instruction": instruction, "input": None}
    prompt = generate_prompt(data_point, LANG_CODES[language])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=1 if decoding_strategy in ["greedy", "multinomial"] else num_beams,
        do_sample=decoding_strategy not in ["greedy", "beam-search"],
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_length,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split(f"### {RESPONSE[LANG_CODES[language]]}:")[1].strip()


def generate(
    instruction,
    language,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_new_tokens=256,
    generation_mode="multinomial",
    **kwargs,
):
    model = load_lora_model(
        base_models[BASE_MODELS[0]], BASE_LORA_MODELS[BASE_MODELS[0]][language]
    )
    data_point = {"instruction": instruction, "input": None}
    prompt = generate_prompt(data_point, "en")
    for x in test_GPT_unconstrained(
        tokenizer,
        model,
        prompt,
        language="English",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        generation_mode=generation_mode,
    ):
        yield x


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.components.Textbox(
            lines=2,
            label="Instruction",
            placeholder="Tell me about alpacas.",
            info="Describe the task the model should perform.",
        ),
        gr.components.Radio(
            value="Multilingual",
            choices=[
                "Multilingual",
                "English",
                "Portuguese",
                "Spanish",
                "Catalan",
                "Basque",
                "Galician",
                "Asturian",
            ],
            label="Model Language",
            interactive=True,
            info="Select fine-tuning language for the model.",
        ),
        gr.components.Slider(
            minimum=0,
            maximum=1,
            value=0.9,
            label="Temperature",
            info="Temperature for Multinomial Sampling[0.1, 1.0].",
        ),
        gr.components.Slider(
            minimum=0,
            maximum=1,
            value=0.75,
            label="Top p",
            info="Top_p probability for Multinomial Sampling [0.1, 1.0]",
        ),
        gr.components.Slider(
            minimum=0,
            maximum=100,
            step=1,
            value=40,
            label="Top k",
            info="Top_k tokens for Multinomial Sampling. [1, 200]",
        ),
        gr.components.Slider(
            minimum=30,
            maximum=1024,
            value=256,
            step=8,
            label="Max length",
            info="Total numbers of tokens (prompt + generated). Use with care, "
            "a large value will be slower [30, 1024]",
        ),
        gr.components.Radio(
            choices=["multinomial", "greedy"],
            label="Decoding strategy",
            value="multinomial",
            info="Greedy decoding is fast and produces deterministic results, "
            "while multinomial sampling allows for more diverse generation more similar to human writing. ",
        ),
    ],
    outputs=[
        gr.components.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="🦙🌲 Alpaca-LoRA",
    description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
    examples=[
        example for example_list in EXAMPLES.values() for example in example_list
    ],
)
demo.queue(concurrency_count=1)
demo.launch(debug=True)

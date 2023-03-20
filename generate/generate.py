import torch
from peft import PeftModel
import transformers
import gradio as gr

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

LANG_CODES = {
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "Catalan": "ca",
    "Basque": "eu",
    "Galician": "gl",
}

PROMPTS = {
    "en": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    "pt": {
        "prompt_input": (
            "Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. "
            "Escreva uma resposta que complete adequadamente o pedido.\n\n"
            "### Instrução:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:"
        ),
        "prompt_no_input": (
            "Abaixo está uma instrução que descreve uma tarefa. "
            "Escreva uma resposta que complete adequadamente o pedido.\n\n"
            "### Instrução:\n{instruction}\n\n### Resposta:"
        ),
    },
    "es": {
        "prompt_input": (
            "A continuación se muestra una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. "
            "Escribe una respuesta que complete adecuadamente la petición.\n\n"
            "### Instrucción:\n{instruction}\n\n### Entrada:\n{input}\n\n### Respuesta:"
        ),
        "prompt_no_input": (
            "A continuación se muestra una instrucción que describe una tarea. "
            "Escribe una respuesta que complete adecuadamente la petición.\n\n"
            "### Instrucción:\n{instruction}\n\n### Respuesta:"
        ),
    },
    "ca": {
        "prompt_input": (
            "A continuació es mostra una instrucció que descriu una tasca, juntament amb una entrada que proporciona més context. "
            "Escriviu una resposta que completi adequadament la petició.\n\n"
            "### Instrucció:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:"
        ),
        "prompt_no_input": (
            "A continuació es mostra una instrucció que descriu una tasca. "
            "Escriviu una resposta que completi adequadament la petició.\n\n"
            "### Instrucció:\n{instruction}\n\n### Resposta:"
        ),
    },
    "eu": {
        "prompt_input": (
            "Azpian ataza bat deskribatzen duen instruzio bat dago, testuinguru gehiago ematen duen sarrera batekin batera. "
            "Idatzi eskaera behar bezala betetzen duen erantzuna.\n\n"
            "### Instrukzioa:\n{instruction}\n\n### Sarrera:\n{input}\n\n### Erantzuna:"
        ),
        "prompt_no_input": (
            "Azpian ataza bat deskribatzen duen instruzio bat dago. "
            "Idatzi eskaera behar bezala betetzen duen erantzuna.\n\n"
            "### Instrukzioa:\n{instruction}\n\n### Erantzuna:"
        ),
    },
    "gl": {
        "prompt_input": (
            "A seguinte é unha instrución que describe unha tarefa, xunto cunha entrada que proporciona máis contexto. "
            "Escriba unha resposta que complete correctamente a solicitude.\n\n"
            "### Instrución:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:"
        ),
        "prompt_no_input": (
            "A seguinte é unha instrución que describe unha tarefa. "
            "Escriba unha resposta que complete correctamente a solicitude.\n\n"
            "### Instrución:\n{instruction}\n\n### Resposta:"
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
}

EXAMPLES = {
    "en": [
        ["Tell me about alpacas.", None, "English"],
        ["Tell me about the president of Mexico in 2019.", None, "English"],
        ["Tell me about the king of France in 2019.", None, "English"],
        ["List all Canadian provinces in alphabetical order.", None, "English"],
        ["Write a Python program that prints the first 10 Fibonacci numbers.", None, "English"],
        [
            "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.", None, "English"
        ],
        ["Tell me five words that rhyme with 'shock'.", None, "English"],
        ["Translate the sentence 'I have no mouth but I must scream' into Spanish.", None, "English"],
    ],
    "pt": [
        ["Conte-me sobre alpacas.", None, "Portuguese"],
        ["Conte-me sobre o presidente do México em 2019.", None, "Portuguese"],
        ["Conte-me sobre o rei da França em 2019.", None, "Portuguese"],
        ["Liste todas as províncias canadenses em ordem alfabética.", None, "Portuguese"],
        [
            "Escreva um programa Python que imprima os primeiros 10 números de Fibonacci.", None, "Portuguese"
        ],
        [
            "Escreva um programa que imprima os números de 1 a 100. Mas para múltiplos de três imprima 'Fizz' em vez do número e para múltiplos de cinco imprima 'Buzz'. Para números que são múltiplos de três e cinco imprima ' FizzBuzz'.", None, "Portuguese"
        ],
        ["Diga-me cinco palavras que rimam com 'choque'.", None, "Portuguese"],
        ["Traduza a frase 'Não tenho boca, mas devo gritar' para o espanhol.", None, "Portuguese"],
    ],
    "es": [
        ["Háblame de las alpacas.", None, "Spanish"],
        ["Háblame del presidente de México en 2019.", None, "Spanish"],
        ["Háblame del rey de Francia en 2019.", None, "Spanish"],
        ["Lista de todas las provincias canadienses en orden alfabético.", None, "Spanish"],
        [
            "Escribe un programa en Python que imprima los primeros 10 números de Fibonacci.", None, "Spanish"
        ],
        [
            "Escriba un programa que imprima los números del 1 al 100. Pero para los múltiplos de tres, escriba 'Fizz' en lugar del número y para los múltiplos de cinco, escriba 'Buzz'. Para los números que son múltiplos de tres y de cinco, escriba ' FizzBuzz'.", None, "Spanish"
        ],
        ["Dime cinco palabras que rimen con 'choque'.", None, "Spanish"],
        ["Traducir la oración 'No tengo boca pero debo gritar' al inglés.", None, "Spanish"],
    ],
    "ca": [
        ["Parla'm de les alpaques.", None, "Catalan"],
        ["Parla'm del president de Mèxic el 2019.", None, "Catalan"],
        ["Parla'm del rei de França el 2019.", None, "Catalan"],
        ["Llista totes les províncies canadenques per ordre alfabètic.", None, "Catalan"],
        [
            "Escriu un programa Python que imprimeixi els primers 10 nombres de Fibonacci.", None, "Catalan"
        ],
        [
            "Escriu un programa que imprimeixi els números de l'1 al 100. Però per a múltiples de tres escriviu 'Fizz' en comptes del nombre i per als múltiples de cinc imprimiu 'Buzz'. Per als números que són múltiples de tres i cinc imprimiu' FizzBuzz'.", None, "Catalan"
        ],
        ["Digues-me cinc paraules que rimin amb 'xoc'.", None, "Catalan"],
        ["Tradueix la frase 'No tinc boca, però he de cridar' al castellà.", None, "Catalan"],
    ],
    "eu": [
        ["Hitz egin alpakei buruz.", None, "Basque"],
        ["Hitz egin 2019ko Mexikoko presidenteaz.", None, "Basque"],
        ["Hitz egin 2019ko Frantziako erregeaz.", None, "Basque"],
        ["Zerrendatu Kanadako probintzia guztiak ordena alfabetikoan.", None, "Basque"],
        ["Idatzi lehen 10 Fibonacci zenbakiak inprimatzen dituen Python programa bat.", None, "Basque"],
        [
            "Idatzi 1etik 100era bitarteko zenbakiak inprimatzen dituen programa bat. Baina hiruren multiploetarako inprimatu 'Fizz' zenbakiaren ordez eta bosten multiploetarako inprimatu 'Buzz'. Hiru eta bosten multiploak diren zenbakietarako inprimatu ' FizzBuzz'.", None, "Basque"
        ],
        ["Esan 'deskarga' hitzaren errima duten bost hitz.", None, "Basque"],
        ["Itzuli 'Ez dut ahorik baina garrasi egin behar dut' esaldia gaztelaniara.", None, "Basque"],
    ],
    "gl": [
        ["Fálame de alpacas.", None, "Galician"],
        ["Fálame do presidente de México en 2019.", None, "Galician"],
        ["Fálame do rei de Francia en 2019.", None, "Galician"],
        ["Enumere todas as provincias canadenses por orde alfabética.", None, "Galician"],
        [
            "Escribe un programa Python que imprima os primeiros 10 números de Fibonacci.", None, "Galician"
        ],
        [
            "Escribe un programa que imprima os números do 1 ao 100. Pero para múltiplos de tres escribe 'Fizz' en lugar do número e para os múltiplos de cinco imprime 'Buzz'. Para números que son múltiplos de tres e cinco imprime ' FizzBuzz'.", None, "Galician"
        ],
        ["Dime cinco palabras que riman con 'choque'.", None, "Galician"],
        ["Traducir ao castelán a frase 'Non teño boca pero debo berrar'.", None, "Galician"],
    ],
}

BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_MODEL = "HiTZ/alpaca-lora-7b"

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
        lora_model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
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


base_model, lora_model, tokenizer = load_model_tokenizer(BASE_MODEL, LORA_MODEL)


def generate_prompt(data_point, lang):
    if data_point["input"]:
        return PROMPTS[lang]["prompt_input"].format_map(data_point)
    else:
        return PROMPTS[lang]["prompt_no_input"].format_map(data_point)


def evaluate(
    instruction,
    input,
    language,
    model_name,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_length=256,
    num_beams=4,
    decoding_strategy="multinomial",
    **kwargs,
):
    global base_model
    global lora_model
    if model_name == BASE_MODEL:
        model = base_model
    elif model_name == LORA_MODEL:
        model = lora_model
    elif "llama" in model_name:
        base_model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = base_model
    elif "lora" in model_name:
        lora_model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
        model = lora_model
    data_point = {"instruction": instruction, "input": input}
    prompt = generate_prompt(data_point, LANG_CODES[language])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=1
        if decoding_strategy in ["greedy", "multinomial"]
        else num_beams,
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


demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2,
            label="Instruction",
            placeholder="Tell me about alpacas.",
            info="Describe the task the model should perform.",
        ),
        gr.components.Textbox(
            lines=2,
            label="Input",
            placeholder=None,
            info="Optional context or input for the task",
        ),
        gr.components.Radio(
            value="English",
            choices=[
                "English",
                "Portuguese",
                "Spanish",
                "Catalan",
                "Basque",
                "Galician",
            ],
            label="Prompt Language",
            interactive=True,
            info="Select a language for the prompt.",
        ),
        gr.components.Radio(
            value="HiTZ/alpaca-lora-7b",
            choices=[
                "HiTZ/alpaca-lora-7b",
                #"chansung/alpaca-lora-13b",
                #"chansung/alpaca-lora-30b",
                "22h/cabrita-lora-v0-1",
                "decapoda-research/llama-7b-hf",
                #"decapoda-research/llama-13b-hf",
                #"decapoda-research/llama-30b-hf",
                #"decapoda-research/llama-65b-hf",
            ],
            label="Model",
            interactive=True,
            info="Select a model to use for generation. LLaMa models are pretrained-only, LoRA models are fine-tuned on the Alpaca dataset.",
        ),
        gr.components.Slider(
            minimum=0,
            maximum=1,
            value=0.1,
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
        gr.components.Slider(
            minimum=1,
            maximum=4,
            step=1,
            value=4,
            label="Num Beams",
            info="Total numbers of beams for beam-search decoding.",
        ),
        gr.components.Radio(
            choices=["multinomial", "greedy", "beam-search", "beam-search multinomial"],
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
    examples=[example for example_list in EXAMPLES.values() for example in example_list]
)
demo.queue(concurrency_count=1)
demo.launch()

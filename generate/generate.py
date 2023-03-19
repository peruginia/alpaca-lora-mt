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
            "### Instrução:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
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

EXAMPLES = {
    "en": [
        ["Tell me about alpacas."],
        ["Tell me about the president of Mexico in 2019."],
        ["Tell me about the king of France in 2019."],
        ["List all Canadian provinces in alphabetical order."],
        ["Write a Python program that prints the first 10 Fibonacci numbers."],
        [
            "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'."
        ],
        ["Tell me five words that rhyme with 'shock'."],
        ["Translate the sentence 'I have no mouth but I must scream' into Spanish."],
    ],
    "pt": [
        ["Conte-me sobre alpacas."],
        ["Conte-me sobre o presidente do México em 2019."],
        ["Conte-me sobre o rei da França em 2019."],
        ["Liste todas as províncias canadenses em ordem alfabética."],
        [
            "Escreva um programa Python que imprima os primeiros 10 números de Fibonacci."
        ],
        [
            "Escreva um programa que imprima os números de 1 a 100. Mas para múltiplos de três imprima 'Fizz' em vez do número e para múltiplos de cinco imprima 'Buzz'. Para números que são múltiplos de três e cinco imprima ' FizzBuzz'."
        ],
        ["Diga-me cinco palavras que rimam com 'choque'."],
        ["Traduza a frase 'Não tenho boca, mas devo gritar' para o espanhol."],
    ],
    "es": [
        ["Háblame de las alpacas."],
        ["Háblame del presidente de México en 2019."],
        ["Háblame del rey de Francia en 2019."],
        ["Lista de todas las provincias canadienses en orden alfabético."],
        [
            "Escribe un programa en Python que imprima los primeros 10 números de Fibonacci."
        ],
        [
            "Escriba un programa que imprima los números del 1 al 100. Pero para los múltiplos de tres, escriba 'Fizz' en lugar del número y para los múltiplos de cinco, escriba 'Buzz'. Para los números que son múltiplos de tres y de cinco, escriba ' FizzBuzz'."
        ],
        ["Dime cinco palabras que rimen con 'choque'."],
        ["Traducir la oración 'No tengo boca pero debo gritar' al inglés."],
    ],
    "ca": [
        ["Parla'm de les alpaques."],
        ["Parla'm del president de Mèxic el 2019."],
        ["Parla'm del rei de França el 2019."],
        ["Llista totes les províncies canadenques per ordre alfabètic."],
        [
            "Escriu un programa Python que imprimeixi els primers 10 nombres de Fibonacci."
        ],
        [
            "Escriu un programa que imprimeixi els números de l'1 al 100. Però per a múltiples de tres escriviu 'Fizz' en comptes del nombre i per als múltiples de cinc imprimiu 'Buzz'. Per als números que són múltiples de tres i cinc imprimiu' FizzBuzz'."
        ],
        ["Digues-me cinc paraules que rimin amb 'xoc'."],
        ["Tradueix la frase 'No tinc boca, però he de cridar' al castellà."],
    ],
    "eu": [
        ["Hitz egin alpakei buruz."],
        ["Hitz egin 2019ko Mexikoko presidenteaz."],
        ["Hitz egin 2019ko Frantziako erregeaz."],
        ["Zerrendatu Kanadako probintzia guztiak ordena alfabetikoan."],
        ["Idatzi lehen 10 Fibonacci zenbakiak inprimatzen dituen Python programa bat."],
        [
            "Idatzi 1etik 100era bitarteko zenbakiak inprimatzen dituen programa bat. Baina hiruren multiploetarako inprimatu 'Fizz' zenbakiaren ordez eta bosten multiploetarako inprimatu 'Buzz'. Hiru eta bosten multiploak diren zenbakietarako inprimatu ' FizzBuzz'."
        ],
        ["Esan 'deskarga' hitzaren errima duten bost hitz."],
        ["Itzuli 'Ez dut ahorik baina garrasi egin behar dut' esaldia gaztelaniara."],
    ],
    "gl": [
        ["Fálame de alpacas."],
        ["Fálame do presidente de México en 2019."],
        ["Fálame do rei de Francia en 2019."],
        ["Enumere todas as provincias canadenses por orde alfabética."],
        [
            "Escribe un programa Python que imprima os primeiros 10 números de Fibonacci."
        ],
        [
            "Escribe un programa que imprima os números do 1 ao 100. Pero para múltiplos de tres escribe 'Fizz' en lugar do número e para os múltiplos de cinco imprime 'Buzz'. Para números que son múltiplos de tres e cinco imprime ' FizzBuzz'."
        ],
        ["Dime cinco palabras que riman con 'choque'."],
        ["Traducir ao castelán a frase 'Non teño boca pero debo berrar'."],
    ],
}

BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "HiTZ/alpaca-lora-7b"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(data_point, lang):
    if data_point["input"]:
        return PROMPTS[lang]["prompt_input"].format_map(data_point)
    else:
        return PROMPTS[lang]["prompt_no_input"].format_map(data_point)


model.eval()
"""
if torch.__version__ >= "2":
    model = torch.compile(model)
"""


def evaluate(
    base_model,
    lora_model,
    language,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    **kwargs,
):
    data_point = {"instruction": instruction, "input": input}
    prompt = generate_prompt(data_point, LANG_CODES[language])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Radio(
            value="decapoda-research/llama-7b-hf",
            choices=[
                "decapoda-research/llama-7b-hf",
            ],
            label="Base Model",
            interactive=True,
        ),
        gr.components.Radio(
            value="HiTZ/alpaca-lora-7b",
            choices=[
                "HiTZ/alpaca-lora-7b",
            ],
            label="LoRA Model",
            interactive=True,
        ),
        gr.components.Radio(
            value="HiTZ/alpaca-lora-7b",
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
        ),
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder=None),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
    ],
    outputs=[
        gr.components.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="🦙🌲 Alpaca-LoRA",
    description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
    examples=EXAMPLES["en"],
)
demo.queue(concurrency_count=1)
demo.launch(server_name="0.0.0.0")

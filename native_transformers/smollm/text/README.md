# SmolLM family
## Table of Contents
1. [SmolLM3](#smollm3)
2. [SmolLM2](#smollm2)
3. [Usage](#usage)
    - [Transformers](#transformers)
    - [Chat in TRL](#chat-in-trl)
    - [Local inference](#local-inference)
    - [Smol-tools](#smol-tools)
4. [Pretraining](#pretraining)
5. [Finetuning and Post-training](#finetuning-and-post-training)
6. [Evaluation](#evaluation)
7. [Data](#data)

# SmolLM3
![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/bUYixmNnbbeYN2tzMLQ9i.png)

SmolLM3 is a 3B parameter language model designed to push the boundaries of small models. It supports dual mode reasoning, 6 languages and long context. SmolLM3 is a fully open model that offers strong performance at the 3B–4B scale.

- [SmolLM3-3B-Base](https://hf.co/HuggingFaceTB/SmolLM3-3B-Base)
- [SmolLM3-3B](https://hf.co/HuggingFaceTB/SmolLM3-3B)
- [Intermediate checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)
- [Blog](https://hf.co/blog/smollm3)

Summary:
- **3B model** trained on 11T tokens, SoTA at the 3B scale and competitive with 4B models
- **Fully open model**, open weights + full training details including public data mixture and training configs
- **Instruct model** with **dual mode reasoning,** supporting think/no_think modes
- **Multilingual support** for 6 languages: English, French, Spanish, German, Italian, and Portuguese
- **Long context** up to 128k with NoPE and using YaRN

# SmolLM2
SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device. You can find our most capable model **🤏 SmolLM2-1.7B-Instruct** [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

In this section you can find everything related to the training of the SmolLM family: SmolLM, SmolLM2 and SmolLM3. This includes pretraining and finetuning code, data curation as well as evaluation. We also recommend [SmolCourse](https://github.com/huggingface/smol-course) for more resources on smol models and how to leverage SmolLM models.

- [Model](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Paper](https://huggingface.co/papers/2502.02737)
- [Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)

## Usage
Our most powerful model is `SmolLM3-3B`, which you can use as an assistant with `transformers`, `vllm`, `trl`, or using quantized versions with tools like `llama.cpp`, `MLX`, and `transformers.js`. For lighter applications, you can also use the smaller models `SmolLM2-360M` and`SmolLM2-135M`, which are suitable for on-device usage and can be integrated similarly.
All available in the [SmolLM3 collection](https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23) and [SmolLM2 collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

For model details, please refer to the model cards.
### Transformers
```bash
pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B" # or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

SmolLM3 supports a dual mode reasoning. We enable extended thinking by default, so the example above generates the output with a reasoning trace. For choosing between enabling, you can provide the `/think` and `/no_think` flags through the system prompt as shown in the snippet below for extended thinking disabled. The code for generating the response with extended thinking would be the same except that the system prompt should have `/think` instead of `/no_think`.

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

We also provide the option of specifying the whether to use extended thinking through the `enable_thinking` kwarg as in the example below. You do not need to set the `/no_think` or `/think` flags through the system prompt if using the kwarg, but keep in mind that the flag in the system prompt overwrites the setting in the kwarg.

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

### Chat in TRL
You can also use the TRL CLI to chat with the model from the terminal:
```bash
pip install trl
trl chat --model_name_or_path HuggingFaceTB/SmolLM2-1.7B-Instruct --device cpu
```

You can find more details on how to leverage the model for use cases such as text summarization, text rewriting and function calling in the model card: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct 

### Local inference
You can use the models locally with frameworks like `llama.cpp`, `MLX`, `MLC` and `transformers.js`. You can find the instructions to run SmolLM2 with these frameworks at [local-inference](../tools/smollm_local_inference/README.md).

### Smol-tools
A collection of lightweight AI-powered tools built with LLaMA.cpp and small language models. These tools are designed to run locally on your machine without requiring expensive GPU resources.
Further instructions on how to use the tools can be found in the [smol-tools README](../tools/smol_tools/README.md).

## Pretraining
You can find scripts for launching pretraining with [nanotron](https://github.com/huggingface/nanotron/) under [pretraining](./pretraining/README.md), we share the exact configs for training SmolLM, SmollM2 and SmollM3. Additionally we provide code for continual-pretraining on SmolLM2 and Llama3.2 3B using nanotron. The SmolLM2 nanotron checkpoints are available [on the hub](https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt) with their optimizer states. The SmolLM3

## Finetuning and Post-training
You can find an example script to finetune SmolLM2 using `TRL` and `PEFT` in the `finetuning` folder. We also link to our post-training scripts for SmolLM2 and SmolLM3 using the alignment handbook. We also recommend [SmolCourse](https://github.com/huggingface/smol-course) for more resources on finetuning smol models and SmolLM2.

## Evaluation

We provide the code for evaluating SmolLM2 and SmolLM3 under [evaluation](./evaluation/README.md).

## Data
We also provide the code for curating the SmolLM datasets in [data](./data/README.md), this includes FineWeb-Edu, FineMath and the [distilabel](https://github.com/argilla-io/distilabel) pipelines for SmolTalk.

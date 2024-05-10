import argparse

import torch
from termcolor import colored
from vllm import SamplingParams

from utils import get_template, read_dataset, load_vllm_model


@torch.no_grad()
def main(model_name: str, chat_template: str, preset: str):
    print(colored(f"model_name: {model_name}", 'blue'))
    template = get_template(chat_template=chat_template, model_name_or_path=model_name)
    print(colored(f"prompt: {template['prompt']}", 'blue'))

    # read dataset.
    dataset = read_dataset(preset)

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
    )

    model = load_vllm_model(model_name)
    template = get_template(chat_template=chat_template, model_name_or_path=model_name)

    formatted_prompts = [template['prompt'].format(instruction=prompt) for prompt in dataset]
    output = model.generate(prompts=formatted_prompts, sampling_params=sampling_params)
    output = [it.outputs[0].text.strip() for it in output]

    for i, (prompt, generated_text) in enumerate(zip(dataset, output)):
        print(colored(f"Example {i + 1}, Prompt: {prompt} ...", "yellow"))
        print(colored(f"[MODEL]: {generated_text}", 'cyan'))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, help="The name of the model to chat with")
    parser.add_argument("--chat_template", type=str, help='The chat template to use', default='hf')
    parser.add_argument("--preset", type=str, help='The preset to evaluate the model.', default='commonsense')

    args = parser.parse_args()
    main(args.model_name, args.chat_template, args.preset)

import argparse
import gc
import random

import torch
from termcolor import colored
from vllm import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils import get_template, read_dataset, load_vllm_model


@torch.no_grad()
def main(model_name_a: str, model_name_b: str, chat_template_a: str, chat_template_b: str, preset: str):
    print(colored(f"model_name: {model_name_a} vs {model_name_b}", 'blue'))

    dataset = read_dataset(preset)
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
    )
    # get model_a results first.
    model = load_vllm_model(model_name_a)
    template = get_template(chat_template=chat_template_a, model_name_or_path=model_name_a)

    formatted_prompts = [template['prompt'].format(instruction=prompt) for prompt in dataset]
    output_a = model.generate(prompts=formatted_prompts, sampling_params=sampling_params)
    output_a = [it.outputs[0].text.strip() for it in output_a]

    del model
    torch.cuda.empty_cache()
    gc.collect()
    destroy_model_parallel()

    # get model_b results.
    model = load_vllm_model(model_name_b)
    template = get_template(chat_template=chat_template_b, model_name_or_path=model_name_b)
    formatted_prompts = [template['prompt'].format(instruction=prompt) for prompt in dataset]
    output_b = model.generate(prompts=formatted_prompts, sampling_params=sampling_params)
    output_b = [it.outputs[0].text.strip() for it in output_b]

    del model
    torch.cuda.empty_cache()
    gc.collect()
    destroy_model_parallel()

    # compare the outputs.
    vote_a_or_b = []
    for prompt, a, b in zip(dataset, output_a, output_b):
        a_is_displayed_first = random.choice([True, False])
        if a_is_displayed_first:
            print(colored(f"Prompt: {prompt}", "yellow"))
            print(colored(f"[MODEL 1]: {a}", 'cyan'))
            print(colored(f"[MODEL 2]: {b}", 'magenta'))
            vote = input(colored("Which model is better? (1/2/x): ", "yellow"))
            if vote == 'x':
                vote_a_or_b.append('tie')
            else:
                vote_a_or_b.append('a' if vote == '1' else 'b')
        else:
            print(colored(f"Prompt: {prompt}", "yellow"))
            print(colored(f"[MODEL 1]: {b}", 'cyan'))
            print(colored(f"[MODEL 2]: {a}", 'magenta'))
            vote = input(colored("Which model is better? (1/2/x): ", "yellow"))
            if vote == 'x':
                vote_a_or_b.append('tie')
            else:
                vote_a_or_b.append('b' if vote == '1' else 'a')

    print(colored(f"Votes: {vote_a_or_b}", "blue"))
    a_count = vote_a_or_b.count('a')
    b_count = vote_a_or_b.count('b')
    tie_count = vote_a_or_b.count('tie')
    print(colored(f"Model A: {a_count}, Model B: {b_count}, Tie: {tie_count}", "blue"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, nargs='+', help="The name of the models to compare")
    parser.add_argument("--chat_template", type=str, nargs='+', help='The chat templates to use', default=['hf', 'hf'])
    parser.add_argument("--preset", type=str, help='The preset to evaluate the model.', default='commonsense')

    args = parser.parse_args()
    main(args.model_name[0], args.model_name[1], args.chat_template[0], args.chat_template[1], args.preset)

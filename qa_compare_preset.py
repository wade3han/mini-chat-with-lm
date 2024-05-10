import argparse
import random

import torch
from termcolor import colored

from utils import get_template, read_dataset, load_hf_model_and_tokenizer


@torch.no_grad()
def main(model_name_a: str, model_name_b: str, chat_template_a: str, chat_template_b: str, preset: str):
    print(colored(f"model_name: {model_name_a} vs {model_name_b}", 'blue'))

    dataset = read_dataset(preset)

    # get model_a results first.
    model, tokenizer = load_hf_model_and_tokenizer(model_name_a)
    template = get_template(chat_template=chat_template_a, model_name_or_path=model_name_a)

    output_a = []
    for i, prompt in enumerate(dataset):
        model_input = template['prompt'].format(instruction=prompt)
        tokenized_model_input = tokenizer(model_input, return_tensors='pt', padding='max_length', max_length=1024,
                                          truncation=True).to('cuda')
        sequence = model.generate(**tokenized_model_input,
                                  do_sample=False,
                                  num_return_sequences=1,
                                  pad_token_id=tokenizer.eos_token_id,
                                  max_new_tokens=1024)
        generated_text = tokenizer.decode(sequence[0][len(tokenized_model_input["input_ids"][0]):],
                                          skip_special_tokens=True)
        output_a.append(generated_text)

    # get model_b results.
    model, tokenizer = load_hf_model_and_tokenizer(model_name_b)
    template = get_template(chat_template=chat_template_b, model_name_or_path=model_name_b)

    output_b = []
    for i, prompt in enumerate(dataset):
        model_input = template['prompt'].format(instruction=prompt)
        tokenized_model_input = tokenizer(model_input, return_tensors='pt', padding='max_length', max_length=1024,
                                          truncation=True).to('cuda')
        sequence = model.generate(**tokenized_model_input,
                                  do_sample=False,
                                  num_return_sequences=1,
                                  pad_token_id=tokenizer.eos_token_id,
                                  max_new_tokens=1024)
        generated_text = tokenizer.decode(sequence[0][len(tokenized_model_input["input_ids"][0]):],
                                          skip_special_tokens=True)
        output_b.append(generated_text)

    # compare the outputs.
    vote_a_or_b = []
    for prompt, a, b in zip(dataset, output_a, output_b):
        a_is_displayed_first = random.choice([True, False])
        if a_is_displayed_first:
            print(colored(f"Prompt: {prompt}", "yellow"))
            print(colored(f"[MODEL 1]: {a}", 'cyan'))
            print(colored(f"[MODEL 2]: {b}", 'magenta'))
            vote = input(colored("Which model is better? (1/2): ", "yellow"))
            vote_a_or_b.append('a' if vote == '1' else 'b')
        else:
            print(colored(f"Prompt: {prompt}", "yellow"))
            print(colored(f"[MODEL 1]: {b}", 'cyan'))
            print(colored(f"[MODEL 2]: {a}", 'magenta'))
            vote = input(colored("Which model is better? (1/2): ", "yellow"))
            vote_a_or_b.append('b' if vote == '1' else 'a')

    print(colored(f"Votes: {vote_a_or_b}", "blue"))
    a_count = vote_a_or_b.count('a')
    b_count = vote_a_or_b.count('b')
    print(colored(f"Model 1: {a_count}, Model 2: {b_count}", "blue"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, nargs='+', help="The name of the models to compare")
    parser.add_argument("--chat_template", type=str, nargs='+', help='The chat templates to use', default=['hf', 'hf'])
    parser.add_argument("--preset", type=str, help='The preset to evaluate the model.', default='commonsense')

    args = parser.parse_args()
    main(args.model_name[0], args.model_name[1], args.chat_template[0], args.chat_template[1], args.preset)

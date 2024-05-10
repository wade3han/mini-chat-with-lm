import argparse

import torch
import transformers
from termcolor import colored

from utils import get_template, read_dataset


@torch.no_grad()
def main(model_name: str, chat_template: str, preset: str):
    print(colored(f"model_name: {model_name}", 'blue'))

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto")
    model = model.cuda()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    template = get_template(chat_template=chat_template, model_name_or_path=model_name)
    print(colored(f"prompt: {template['prompt']}", 'blue'))

    # read dataset.
    dataset = read_dataset(preset)

    for i, prompt in enumerate(dataset):
        print(colored(f"Example {i + 1}, Prompt: {prompt} ...", "yellow"))
        model_input = template['prompt'].format(instruction=prompt)
        tokenized_model_input = tokenizer(model_input, return_tensors='pt', padding='max_length', max_length=1024,
                                          truncation=True).to('cuda')
        sequence = model.generate(**tokenized_model_input,
                                  do_sample=False,
                                  num_return_sequences=1,
                                  pad_token_id=tokenizer.eos_token_id,
                                  max_new_tokens=1024)
        generated_text = tokenizer.decode(sequence[0], skip_special_tokens=True)
        generated_text = generated_text[len(model_input):]

        print(colored(f"[MODEL]: {generated_text}", 'cyan'))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, help="The name of the model to chat with")
    parser.add_argument("--chat_template", type=str, help='The chat template to use', default='hf')
    parser.add_argument("--preset", type=str, help='The preset to evaluate the model.', default='commonsense')

    args = parser.parse_args()
    main(args.model_name, args.chat_template, args.preset)
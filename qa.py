import argparse

import torch
import transformers
from termcolor import colored

from template import get_template


@torch.no_grad()
def main(model_name: str):
    print(colored(f"model_name: {model_name}", 'blue'))

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = get_template(model_name)
    print(colored(f"prompt: {template['prompt']}", 'blue'))
    print('\n\n')
    print(colored("Chat with the model (type 'exit' to quit, 'clear' to clear the screen):", 'green'))
    # print the chat template and the system prompt

    chat_history = []
    while True:
        user_input = input(colored("[USER]: ", 'yellow'))
        if user_input == "exit":
            break
        elif user_input == "clear":
            # Clear the screen in a cross-platform way
            print("\033[H\033[J", end="")
            continue

        model_input = template['prompt'].format(instruction=user_input)
        tokenized_model_input = tokenizer(model_input, return_tensors='pt', padding='max_length', max_length=1024,
                                          truncation=True)
        sequence = model.generate(**tokenized_model_input,
                                  temperature=0.0,
                                  num_return_sequences=1,
                                  max_new_tokens=1024)
        generated_text = tokenizer.decode(sequence[0], skip_special_tokens=True)

        # Append the user prompt and the model's response to the chat history
        chat_history.append((user_input, generated_text))

        # Print the model's response
        print(colored("[MODEL]: " + generated_text, 'cyan'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, help="The name of the model to chat with")

    args = parser.parse_args()
    main(args.model_name)

import argparse

import torch
import transformers
from termcolor import colored

from template import get_template


@torch.no_grad()
def main(model_name: str):
    print(colored(f"model_name: {model_name}", 'blue'))

    pipeline = transformers.pipeline("text-generation",
                                     model=model_name,
                                     trust_remote_code=True,
                                     torch_dtype=torch.bfloat16,
                                     device_map="auto")

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
        sequence = pipeline(model_input, temperature=1.0, num_return_sequences=1, max_new_tokens=64)
        generated_text = sequence[0]['generated_text'][len(model_input):]

        # Append the user prompt and the model's response to the chat history
        chat_history.append((user_input, generated_text))

        # Print the model's response
        print(colored("[MODEL]: " + generated_text, 'cyan'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument("--model_name", type=str, help="The name of the model to chat with")

    args = parser.parse_args()
    main(args.model_name)

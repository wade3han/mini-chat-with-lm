import re

from transformers import AutoTokenizer

DEFAULT_PROMPT = {
    "description": "Default template",
    "prompt": "{instruction}",
}

ALPACA_PROMPT = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
}
VICUNA_1_0_PROMPT = {
    "description": "Template used by Vicuna 1.0 and stable vicuna.",
    "prompt": "### Human: {instruction}\n### Assistant:",
}

VICUNA_PROMPT = {
    "description": "Template used by Vicuna.",
    "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
}

OASST_PROMPT = {
    "description": "Template used by Open Assistant",
    "prompt": "<|prompter|>{instruction}<|endoftext|><|assistant|>"
}
OASST_PROMPT_v1_1 = {
    "description": "Template used by newer Open Assistant models",
    "prompt": "<|prompter|>{instruction}</s><|assistant|>"
}

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_PROMPT = {
    "description": "Template used by Llama2 Chat",
    "prompt": "[INST] <<SYS>>\n" + LLAMA2_DEFAULT_SYSTEM_PROMPT + "\n<</SYS>>\n\n{instruction} [/INST] "
}

INTERNLM_PROMPT = {  # https://github.com/InternLM/InternLM/blob/main/tools/alpaca_tokenizer.py
    "description": "Template used by INTERNLM-chat",
    "prompt": "<|User|>:{instruction}<eoh><|Bot|>:"
}

KOALA_PROMPT = {  # https://github.com/young-geng/EasyLM/blob/main/docs/koala.md#koala-chatbot-prompts
    "description": "Template used by EasyLM/Koala",
    "prompt": "BEGINNING OF CONVERSATION: USER: {instruction} GPT:"
}

# Get from Rule-Following: cite
FALCON_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

MPT_PROMPT = {  # https://huggingface.co/TheBloke/mpt-30B-chat-GGML
    "description": "Template used by MPT",
    "prompt": '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n''',
}

CHATGLM_PROMPT = {  # https://github.com/THUDM/ChatGLM-6B/issues/124
    "description": "Template used by ChatGLM",
    "prompt": "问: {instruction}\n答:"
}

DOLLY_PROMPT = {
    "description": "Template used by Dolly",
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
}

TULU2_PROMPT = {
    "description": "Template used by Tulu2 SFT and DPO",
    "prompt": "<|user|>\n{instruction}\n<|assistant|>\n",
}

OLMO_CHAT_PROMPT = {
    "description": "Template used by OLMo Chat models SFT and DPO",
    "prompt": "<|user|>\n{instruction}\n<|assistant|>\n",
}

OPENAI_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
}

LLAMA2_70B_OASST_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

FALCON_INSTRUCT_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

FALCON_CHAT_PROMPT = {  # https://huggingface.co/blog/falcon-180b#prompt-format
    "description": "Template used by Falcon Chat",
    "prompt": "User: {instruction}\nFalcon:",
}

ORCA_2_PROMPT = {
    "description": "Template used by microsoft/Orca-2-13b",
    "prompt": "<|im_start|>system\nYou are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"
}

MISTRAL_PROMPT = {
    "description": "Template used by Mistral Instruct",
    "prompt": "[INST] {instruction} [/INST]"
}

BAICHUAN_CHAT_PROMPT = {
    "description": "Template used by Baichuan2-chat",
    "prompt": "<reserved_106>{instruction}<reserved_107>"
}

QWEN_CHAT_PROMPT = {
    "description": "Template used by Qwen-chat models",
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

ZEPHYR_ROBUST_PROMPT = {
    "description": "",
    "prompt": "<|user|>\n{instruction}</s>\n<|assistant|>\n"
}

MIXTRAL_PROMPT = {
    "description": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "[INST] {instruction} [/INST]"
}


########## CHAT TEMPLATE ###########

def get_template(model_name_or_path=None, system_message=None):
    _model_name_or_path = model_name_or_path.lower() if model_name_or_path else ""
    template = None

    # ===== Check for some older chat model templates ====
    if "wizard" in _model_name_or_path:
        template = VICUNA_PROMPT
    elif "vicuna" in _model_name_or_path:
        template = VICUNA_PROMPT
    elif "oasst" in _model_name_or_path or "openassistant" in _model_name_or_path:
        if "llama2-13b-orca" in model_name_or_path:
            template = OASST_PROMPT_v1_1
        elif "llama2-70b-oasst" in model_name_or_path:
            template = LLAMA2_70B_OASST_CHATML_PROMPT
        else:
            template = OASST_PROMPT
    elif re.search("llama-2-\d+b-chat",
                   _model_name_or_path) or "llama-2-chat" in _model_name_or_path or "llama-2-weights-" in _model_name_or_path:
        template = LLAMA2_CHAT_PROMPT
    elif re.search("llama-2-\d+b", _model_name_or_path):
        template = DEFAULT_PROMPT
    elif re.search("falcon-\d+b-instruct", _model_name_or_path):
        template = FALCON_INSTRUCT_PROMPT
    elif re.search("falcon-\d+b-chat", _model_name_or_path):
        template = FALCON_CHAT_PROMPT
    elif re.search("mpt-\d+b-chat", _model_name_or_path):
        template = MPT_PROMPT
    elif "koala" in _model_name_or_path:
        template = KOALA_PROMPT
    elif "chatglm" in _model_name_or_path:
        template = CHATGLM_PROMPT
    elif "dolly" in _model_name_or_path:
        template = DOLLY_PROMPT
    elif "internlm" in _model_name_or_path:
        template = INTERNLM_PROMPT
    elif re.search("mistral-\d+b-instruct", _model_name_or_path) or re.search("mixtral.-instruct", _model_name_or_path):
        template = MISTRAL_PROMPT
    elif re.search("orca-2-\d+b", _model_name_or_path):
        template = ORCA_2_PROMPT
    elif "baichuan2" in _model_name_or_path:
        template = BAICHUAN_CHAT_PROMPT
    elif re.search("qwen-\d+b-chat", _model_name_or_path):
        template = QWEN_CHAT_PROMPT
    elif "zephyr_7b_robust" in _model_name_or_path:
        template = ZEPHYR_ROBUST_PROMPT
    elif re.search("tulu2_dpo_\d+b", _model_name_or_path):
        template = TULU2_PROMPT
    elif re.search("tulu2_\d+b", _model_name_or_path):
        template = TULU2_PROMPT
    elif re.search("olmo_\d+b_finetune", _model_name_or_path) or \
            re.search("olmo_\d+b_finetune_dpo", _model_name_or_path):
        template = OLMO_CHAT_PROMPT
    else:
        # ======== Else default to tokenizer.apply_chat_template =======
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            template = [{'role': 'system', 'content': system_message},
                        {'role': 'user', 'content': '{instruction}'}] if system_message else [
                {'role': 'user', 'content': '{instruction}'}]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Check if the prompt starts with the BOS token
            # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            template = {'description': f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)",
                        'prompt': prompt}

        except Exception as e:
            assert template is not None, \
                f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."

    return template

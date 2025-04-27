import os

from dotenv import load_dotenv

import llamator

# from llamator.attacks.vlm_lowres_docs import TestVlmLowresPdf
from llamator.attacks.vlm_m_attack import TestVlmMAttack

# from llamator.attacks.vlm_text_hallucination import TestVlmTextHallucination
from llamator.client.specific_chat_clients import ClientOpenAI

load_dotenv(".env")


base_url_tested = "https://api.openai.com/v1/"
# base_url_tested = "http://localhost:7113/v1"

model_tested = "gpt-4o-mini"
# model_tested = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_tested = "OpenGVLab/InternVL2_5-8B-MPO"
# model_tested = "unsloth/Llama-3.2-11B-Vision-Instruct"

client_tested = ClientOpenAI(
    api_key=os.environ["OPENAI_API_KEY"], base_url=base_url_tested, model=model_tested  # fmt: skip
)

client_judge = llamator.ClientOpenAI(  # LLM for judging
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1/",
    model="gpt-4o",
    temperature=0.8,
    system_prompts=[],
)

client_attack = ClientOpenAI(
    api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1/", model="gpt-4o"
)


# Test vision client
messages_test_vision = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
        ],
    }
]

# out = client_tested.interact(history=[], messages=messages_test_vision)
# print(out)
# exit()

test_params = [
    # ("vlm_m_attack", {"num_attempts": 3, "attack_source": "huggingface"}),
    # ("vlm_text_hallucination", {"num_attempts": 3}),
    ("vlm_lowres_docs", {"num_attempts": 3}),
    # ("system_prompt_leakage", {"num_attempts": 2, "multistage_depth": 3}),
]

config = {
    "enable_logging": True,  # Enable logging
    "enable_reports": True,  # Enable report generation
    "artifacts_path": "./artifacts",  # Path to the directory for saving artifacts
    "debug_level": 1,  # Logging level: 0 - WARNING, 1 - INFO, 2 - DEBUG
    "report_language": "en",  # Report language: 'en', 'ru'
}

llamator.start_testing(
    attack_model=client_attack,
    tested_model=client_tested,
    judge_model=client_judge,
    config=config,
    basic_tests=test_params,
)

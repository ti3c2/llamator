import os

from dotenv import load_dotenv

import llamator
from llamator.attacks.vlm_lowres_docs import TestVlmLowresPdf
from llamator.attacks.vlm_m_attack import TestVlmMAttack
from llamator.attacks.vlm_text_hallucination import TestVlmTextHallucination
from llamator.client.specific_chat_clients import ClientOpenAI

load_dotenv(".env")


base_url_tested = "http://localhost:7113/v1"
model_tested = "llava-hf/llava-v1.6-mistral-7b-hf"
model_tested = "OpenGVLab/InternVL2_5-8B-MPO"
model_tested = "unsloth/Llama-3.2-11B-Vision-Instruct"

client_tested = ClientOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # fmt: skip
    base_url=base_url_tested,
    model=model_tested
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

tests_with_attempts = [
    # ("complimentary_transition", 3),
    # ("ethical_compliance", 2),
    # ("linguistic_evasion", 2),
    # ("harmful_behavior_multistage", 2),
]

custom_tests_with_attempts = [
    (TestVlmTextHallucination, 100),
    (TestVlmLowresPdf, 100),
    (TestVlmMAttack, 90),
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
    config=config,
    tests_with_attempts=tests_with_attempts,
    custom_tests_with_attempts=custom_tests_with_attempts,
    multistage_depth=3,
    num_threads=3,
)
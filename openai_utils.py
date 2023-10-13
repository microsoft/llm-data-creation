import json
import time
from typing import Any, Dict


def read_json(filepath: str) -> Dict[Any, Any]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


MAX_TOKENS = 2048  # token generation should generate the only one token


def api_query(openai: Any, description: str, text: str, model: str = "gpt-3.5-turbo") -> Any:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": description},
                {"role": "user", "content": text},
            ],
            temperature=1,
            top_p=0,
        )
        return_text = response["choices"][0]["message"]["content"]
        return_usage = response["usage"]["total_tokens"]

    except openai.error.InvalidRequestError:  # number of tokens > 4097
        return_text, return_usage = None, None
    except openai.error.ServiceUnavailableError:  # server overload
        time.sleep(3)
        return_text, return_usage = api_query(openai, description, text, model)
    except openai.error.RateLimitError:  # server overload
        time.sleep(5)
        return_text, return_usage = api_query(openai, description, text, model)
    except openai.error.APIError:  # API error
        time.sleep(5)
        return_text, return_usage = api_query(openai, description, text, model)

    return return_text, return_usage

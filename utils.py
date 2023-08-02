def get_max_tokens(model_name: str) -> int:
    if model_name.startswith("gpt-3.5-turbo-16k"):
        return 16384
    if model_name.startswith("gpt-4-32k"):
        return 32768
    if model_name.startswith("gpt-4"):
        return 8192
    return 4096

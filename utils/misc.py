import tiktoken

def truncate_to_max_tokens(text: str, max_tokens: int, encoding="cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens) 
import pytest
from toolemu.utils.misc import truncate_to_max_tokens, get_num_tokens

MAX_TOKENS = 16384

def test_truncate_to_max_tokens():
    # Create a string that will exceed the max token limit
    base = "hello " * (MAX_TOKENS + 1000)
    orig_tokens = get_num_tokens(base)
    truncated = truncate_to_max_tokens(base, MAX_TOKENS)
    truncated_tokens = get_num_tokens(truncated)
    print(f"Original token count: {orig_tokens}")
    print(f"Truncated token count: {truncated_tokens}")
    assert truncated_tokens <= MAX_TOKENS
    assert truncated != base
    # If the string is already short enough, it should not be truncated
    short = "hello world"
    assert truncate_to_max_tokens(short, MAX_TOKENS) == short 
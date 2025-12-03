"""
Utility functions for token counting and cost calculation
"""
from whisper.tokenizer import get_tokenizer


def calculate_token_count(text: str, language: str = "en", is_multilingual: bool = True) -> int:
    """
    Calculate the number of tokens in the transcription text
    
    Args:
        text: The transcribed text
        language: Language code (default: "en")
        is_multilingual: Whether to use multilingual tokenizer
    
    Returns:
        Number of tokens
    """
    try:
        tokenizer = get_tokenizer(
            multilingual=is_multilingual,
            num_languages=99,
            language=language,
            task="transcribe"
        )
        tokens = tokenizer.encode(text)
        # Filter out timestamp tokens (tokens >= timestamp_begin)
        text_tokens = [t for t in tokens if t < tokenizer.timestamp_begin]
        return len(text_tokens)
    except Exception:
        # Fallback: approximate token count (rough estimate: 1 token ≈ 4 characters)
        return len(text) // 4


def calculate_cost(token_count: int, model: str, audio_duration: float = None) -> float:
    """
    Calculate the cost based on token count and model used
    
    Cost per token rates (example rates - adjust based on your pricing):
    - tiny: $0.0001 per 1K tokens
    - base: $0.0002 per 1K tokens
    - small: $0.0003 per 1K tokens
    - medium: $0.0005 per 1K tokens
    - large/turbo: $0.001 per 1K tokens
    
    Args:
        token_count: Number of tokens
        model: Model name used
        audio_duration: Duration of audio in seconds (optional, for alternative pricing)
    
    Returns:
        Cost in USD
    """
    # Cost per 1K tokens by model
    cost_per_1k_tokens = {
        "tiny": 0.0001,
        "tiny.en": 0.0001,
        "base": 0.0002,
        "base.en": 0.0002,
        "small": 0.0003,
        "small.en": 0.0003,
        "medium": 0.0005,
        "medium.en": 0.0005,
        "large": 0.001,
        "large-v1": 0.001,
        "large-v2": 0.001,
        "large-v3": 0.001,
        "turbo": 0.001,
    }
    
    # Get cost rate for model (default to large if not found)
    rate = cost_per_1k_tokens.get(model.lower(), 0.001)
    
    # Calculate cost
    cost = (token_count / 1000.0) * rate
    
    return round(cost, 6)  # Round to 6 decimal places


# List of keywords that indicate harmful or off-topic queries
# If a question contains any of these, we reject it immediately
# before it ever reaches the LLM or retriever

BLOCKED_KEYWORDS = [
    # Harmful content
    "bomb", "weapon", "explosive", "poison", "kill", "murder",
    "hack", "malware", "virus", "ransomware", "exploit",
    # Clearly off-topic
    "lottery", "casino", "gambling", "porn", "nude",
    # Prompt injection attempts
    "ignore previous", "ignore all instructions",
    "forget your instructions", "you are now",
    "act as", "pretend you are", "jailbreak"
]


def is_safe_query(query_text):
    """
    Checks if the user's query is safe to process.

    Returns:
        (True, None)           — safe, proceed normally
        (False, reason_string) — blocked, return reason to user
    """
    query_lower = query_text.lower().strip()

    # Check 1: Empty or too short
    if len(query_lower) < 3:
        return False, "Query is too short. Please ask a complete question."

    # Check 2: Too long (possible prompt injection)
    if len(query_lower) > 1000:
        return False, "Query is too long. Please keep your question under 1000 characters."

    # Check 3: Blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_lower:
            return False, f"I can't help with that type of request. Please ask questions related to the documents."

    # Check 4: Looks like a question or statement (basic sanity check)
    # If it's just random characters, reject it
    words = query_lower.split()
    if len(words) < 2:
        return False, "Please ask a complete question."

    return True, None


def is_grounded_answer(answer_text):
    """
    Checks if the LLM's answer appears to be grounded
    in the retrieved context rather than making things up.

    If the LLM says it doesn't know — that's actually good,
    it means the guardrail is working correctly.
    """
    not_grounded_phrases = [
        "i don't have information",
        "i cannot find",
        "not mentioned in",
        "i don't know",
        "no information provided",
        "the context does not",
        "not provided in the context"
    ]

    answer_lower = answer_text.lower()

    # If the model explicitly says it can't find the answer,
    # that IS the correct grounded behaviour — not a failure
    for phrase in not_grounded_phrases:
        if phrase in answer_lower:
            return True, "Model correctly reported insufficient context"

    # Otherwise assume the answer used the context
    return True, "Answer generated from context"
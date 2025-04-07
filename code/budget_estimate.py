import tiktoken
import sys

import tiktoken

def track_usage_and_cost(response, price_per_million_input, price_per_million_output, model, usage=None):
    total_input_tokens = response.prompt_tokens
    total_output_tokens = response.completion_tokens
    cost_input = (total_input_tokens / 1_000_000) * price_per_million_input
    cost_output = (total_output_tokens / 1_000_000) * price_per_million_output
    total_cost = cost_input + cost_output

    if not usage:
        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost_input": round(cost_input, 6),
            "cost_output": round(cost_output, 6),
            "total_cost": round(total_cost, 6)
        }

    else:
        usage["input_tokens"] += total_input_tokens
        usage["output_tokens"] += total_output_tokens
        usage["cost_input"] += cost_input
        usage["cost_output"] += cost_output
        usage["total_cost"] += total_cost
        usage['total_tokens'] += (total_input_tokens + total_output_tokens)




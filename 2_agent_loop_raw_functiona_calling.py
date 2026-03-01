from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (Langchain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "smartphone": 799.99, "headphones": 199.99}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discounts_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discounts_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Lookup the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply (e.g. bronze, silver, gold)"
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages, tools):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }


    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shipping assistant."
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You must call the get_product_price tool first to get the real price of a product.\n"
                "2. Only call apply_discount if you have a real price from get_product_price "
                "and the question requires a discount to be applied.\n"
                "3. Never calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user doesnot specify a discount tier, "
                "ask them which tier to use - do NOT assume one.\n"
            ),
        },
        {"role": "user", "content": question},
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {i} ---")

        response = ollama_chat_traced(messages, tools_for_llm)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, we can stop the loop
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use(**tool_args)
        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)  # Add the AI message with the tool call
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")

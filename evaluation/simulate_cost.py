import tiktoken


class CostSimulation:

    def __init__(self, gpt_model_name: str = "gpt-4o-mini"):
        self.gpt_model_name = gpt_model_name
        self.prices = {
            "gpt-4": {"in": 30.0, "out": 60.0},
            "gpt-4-32k": {"in": 60.0, "out": 120.0},
            "gpt-4o": {"in": 2.5, "out": 10.0},
            "gpt-4o-mini": {"in": 0.15, "out": 0.60},
            "gpt-5": {"in": 1.25, "out": 10.0},
            "gpt-5-mini": {"in": 0.25, "out": 2.0},
            "gpt-5-nano": {"in": 0.05, "out": 0.40},
        }

    def token_count(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.gpt_model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def simulate_cost(self, requests):

        input_price = self.prices[self.gpt_model_name]["in"]
        output_price = self.prices[self.gpt_model_name]["out"]
        total_cost = 0.0
        for req in requests:
            n_tokens = self.token_count(req)

            total_cost += (n_tokens / 1_000_000) * input_price
            total_cost += (70 / 1_000_000) * output_price

        return round(total_cost, 6)

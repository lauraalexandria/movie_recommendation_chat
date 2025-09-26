import tiktoken

# Tenho que pensar em uma forma de usar essa função como uma flag para rodar
# as demais funções e então conferir os valores para decidir de executo ou
# não o comando, possivelmente adicionar if else no for doc in tqdm(documents):
# e assim calcular o valor mínimo ou outra coisa


class CostSimulation:

    def __init__(self, gpt_model_name: str = "gpt-4o-mini"):
        self.gpt_model_name = gpt_model_name

    def token_count(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.gpt_model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def simulate_cost(self, requests, price=0.15):

        total_cost = 0.0
        for req in requests:
            n_tokens = self.token_count(req)

            total_cost += (n_tokens / 1_000_000) * price

        return round(total_cost, 6)

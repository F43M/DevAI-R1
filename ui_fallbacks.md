# Fallbacks de interface

- Quando o DevAI é executado sem frontend web, as dicas de tooltips e o botão "Ajuda" não são exibidos.
- Utilize a opção `--help` na CLI para ver um resumo das funções disponíveis.
- Em execução via terminal, as funções avançadas continuam acessíveis pelo menu textual tradicional.
- Caso a biblioteca `rich` não esteja presente, a CLI entra em modo `--plain` automaticamente.
- # FUTURE: exibir mensagens contextuais a cada comando executado.
- Em telas muito pequenas ou navegadores antigos, os botões são empilhados automaticamente para evitar cortes de texto.

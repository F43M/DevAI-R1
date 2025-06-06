# DevAI-R1
Sistema de inteligência de desenvolvimento com IAs.

Esta atualização permite explorar todo o limite de 160k tokens
oferecido pela API do OpenRouter para entrada e saída de dados.
Agora o aplicativo define `MAX_CONTEXT_LENGTH` em `160000` e envia
prompts para a API utilizando esse limite, garantindo respostas mais
longas quando necessário.

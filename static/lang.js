window.strings = {
  "analyze_button": "🔎 Analisar",
  "project_analysis": "📊 Analisar Projeto",
  "simulate_refactor": "🕶️ Simular Refatoração",
  "apply_refactor": "💾 Aplicar Refatoração",
  "auto_monitor": "🧭 Monitoramento Automático",
  "train_from_errors": "🧠 Aprendizado com Erros"
};

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-string]').forEach(el => {
    const key = el.dataset.string;
    if (window.strings[key]) {
      el.textContent = window.strings[key];
    }
  });
});

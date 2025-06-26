Code Model Fine-tuning
======================

This tutorial explains how to create datasets suitable for fine-tuning code generation models using the GitHub scraping plugin and the utilities provided in :mod:`training`.

1. Run the GitHub plugin to download repositories by topic or user.
   The example below collects the README files of popular projects tagged
   ``machine-learning``::

       from plugins import load_plugin, run_plugin

       plugin = load_plugin("github_scraper")
       records = run_plugin(plugin, ["en"], ["machine-learning"])

   Each record contains the repository name, language and the cleaned text of the README.

2. Save the dataset in the Hugging Face format so it can be loaded directly by ``datasets``::

       from training.pipeline import save_json
       from datasets import Dataset

       save_json(records, "github_code.json")
       ds = Dataset.from_json("github_code.json")
       ds.push_to_hub("my/github-code")  # optional

3. Prepare the inputs for your model. For encoder models use ``prepare_bert_inputs``::

       from training.pretrained_utils import prepare_bert_inputs

       batch = prepare_bert_inputs([r["content"] for r in records])

4. Fine-tune the model with ``transformers``::

       from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

       model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
       args = TrainingArguments(output_dir="ft-code", num_train_epochs=1)
       trainer = Trainer(model=model, args=args, train_dataset=ds)
       trainer.train()

The resulting model can then be used to generate code snippets conditioned on the README text or other prompts.

Example Airflow configuration
----------------------------

The `publish_dataset` task in :mod:`training.airflow_pipeline` can automatically
start a fine-tuning run when the ``FINE_TUNE_MODEL`` environment variable is set.
``MODEL_NAME`` selects the pretrained checkpoint used::

   AIRFLOW_SCHEDULE="@daily"
   FINE_TUNE_MODEL="1"
   MODEL_NAME="bert-base-uncased"

Both the ``dataset_version`` and ``model_version`` are logged to MLflow for
experiment tracking.

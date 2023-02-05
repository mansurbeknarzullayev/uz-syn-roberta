# uz-syn-roberta

## Requirements
<pre>
pip install git+https://github.com/huggingface/transformers
</pre>
[PyTorch](https://pytorch.org/)

## Train
<pre>from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import torch

path = 'dataset.txt'
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=path, vocab_size=52_000, min_frequency=2, special_tokens=["&lt;s&gt;", "&lt&lt;pad&gt;", "&lt&lt;/s&gt;", "&lt&lt;unk&gt;", "&lt&lt;mask&gt;",])
tokenizer.save_model("models/UzRoBERTa")
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
tokenizer = RobertaTokenizerFast.from_pretrained("models/UzRoBERTa", max_len=512)
model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="dataset.txt",
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="models/UzRoBERTa",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("models/UzRoBERTa")</pre>

## Use
<pre>from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="Mansurbek/uz-syn-roberta"
)
print(*fill_mask("Tadbirkorlik – foyda olish &lt&lt;mask&gt; faoliyat."), sep = '\n')
</pre>

## Result
<pre>
{'score': 0.17550185322761536, 'token': 395, 'token_str': ' uchun', 'sequence': 'Tadbirkorlik – foyda olish uchun faoliyat.'}
{'score': 0.03933406248688698, 'token': 298, 'token_str': ' va', 'sequence': 'Tadbirkorlik – foyda olish va faoliyat.'}
{'score': 0.03401805832982063, 'token': 1719, 'token_str': ' maqsadida', 'sequence': 'Tadbirkorlik – foyda olish maqsadida faoliyat.'}
{'score': 0.02175612561404705, 'token': 358, 'token_str': ' bilan', 'sequence': 'Tadbirkorlik – foyda olish bilan faoliyat.'}
{'score': 0.01759626343846321, 'token': 16, 'token_str': ',', 'sequence': 'Tadbirkorlik – foyda olish, faoliyat.'}
</pre>

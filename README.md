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

dataset_file = 'data/dataset.txt'
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=dataset_file, vocab_size=52_000, min_frequency=2, special_tokens=["&lt;s&gt;", "&lt;pad&gt;", "&lt;/s&gt;", "&lt;unk&gt;", "&lt;mask&gt;",])
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
    file_path=dataset_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="models/UzRoBERTa",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
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
trainer.save_model("models/UzRoBERTa")
</pre>

## Use RoBERTa
<pre>from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="Mansurbek/uz-syn-roberta"
)
print(*fill_mask("Tadbirkorlik – foyda olish &lt;mask&gt; faoliyat."), sep = '\n')

</pre>

## Result
<pre>
{'score': 0.3028637170791626, 'token': 401, 'token_str': ' uchun', 'sequence': 'Tadbirkorlik – foyda olish uchun faoliyat.'}
{'score': 0.08473362773656845, 'token': 16, 'token_str': ',', 'sequence': 'Tadbirkorlik – foyda olish, faoliyat.'}
{'score': 0.0782223641872406, 'token': 297, 'token_str': ' va', 'sequence': 'Tadbirkorlik – foyda olish va faoliyat.'}
{'score': 0.06843182444572449, 'token': 1367, 'token_str': ' maqsadida', 'sequence': 'Tadbirkorlik – foyda olish maqsadida faoliyat.'}
{'score': 0.028617164120078087, 'token': 613, 'token_str': ' –', 'sequence': 'Tadbirkorlik – foyda olish – faoliyat.'}
</pre>

## Use Synonymizer
<pre>
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/UzRoBERTa",
    tokenizer="models/UzRoBERTa"
)

synset = []

synset.append(['foyda', 'daromad'])
synset.append(['uchun', 'maqsadida', 'maqsadda'])
synset.append(['olish', 'qilish'])

def get_syn_words(word):
    ret = []
    for v in synset:
        if word in v:
            ret += v
    return ret
def get_syn(s):
    words = s.split()
    ret = []
    for i in range(len(words)):
        word = words[i]
        words[i] = '&lt;mask&gt;'
        for syn in get_syn_words(word):
            for token in fill_mask(' ' . join(words), top_k=100):
                if token['token_str'].strip() == syn and syn != word:
                    ret.append(token)
        words[i] = word
    return ret
sentence = "Tadbirkorlik – foyda olish uchun faoliyat."

print(*get_syn(sentence), sep = '\n')
</pre>

## Result
<pre>
{'score': 0.002073188778012991, 'token': 2683, 'token_str': ' daromad', 'sequence': 'Tadbirkorlik – daromad olish uchun faoliyat.'}
{'score': 0.08348902314901352, 'token': 531, 'token_str': ' qilish', 'sequence': 'Tadbirkorlik – foyda qilish uchun faoliyat.'}
{'score': 0.06843182444572449, 'token': 1367, 'token_str': ' maqsadida', 'sequence': 'Tadbirkorlik – foyda olish maqsadida faoliyat.'}
</pre>

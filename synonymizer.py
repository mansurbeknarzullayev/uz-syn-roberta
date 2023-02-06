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
        words[i] = '<mask>'
        for syn in get_syn_words(word):
            for token in fill_mask(' ' . join(words), top_k=100):
                if token['token_str'].strip() == syn and syn != word:
                    ret.append(token)
        words[i] = word
    return ret
sentence = "Tadbirkorlik – foyda olish uchun faoliyat."

print(*get_syn(sentence), sep = '\n')


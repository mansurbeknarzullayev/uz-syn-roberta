from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="Mansurbek/uz-syn-roberta"
)
print(*fill_mask("Tadbirkorlik – foyda olish <mask> faoliyat."), sep='\n')

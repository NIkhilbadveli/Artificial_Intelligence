"""
Machine Translation using Transformer.
We will use Helsinki-NLP pre-trained transformer models for this task.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def basic_translation(src_lang, dst_lang, text):
    """
    Uses pipeline to translate text from src_lang to dst_lang
    :param src_lang:
    :param dst_lang:
    :param text:
    :return:
    """

    task_name = f"translation_{src_lang}_to_{dst_lang}"  # Task identifier
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dst_lang}"  # Model name identifier for huggingface.co/models

    translator = pipeline(task=task_name, model=model_name, tokenizer=model_name)

    # Translate text
    output = translator(text)
    print('\nTranslated text in German:-')
    print(output[0]['translation_text'])


# The above method is not very flexible since all we get is a pipeline object.

def get_translation_model_and_tokenizer(src_lang, dst_lang):
    """
    Given the source and destination languages, returns the appropriate model
    See the language codes here: https://developers.google.com/admin-sdk/directory/v1/languages
    For the 3-character language codes, you can google for the code!
    """
    # construct our model name
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dst_lang}"
    # initialize the tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # return them for use
    return model, tokenizer


def translate_using_model_tokenizer(model, tokenizer, text):
    """
    Translates text using the model and tokenizer got from above function
    :param model:
    :param tokenizer:
    :param text:
    :return:
    """
    # encode the text
    encoded_text = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    # translate the text
    tsl_text = model.generate(encoded_text)  # This uses greedy decoding
    # decode the text
    tsl_text = tokenizer.decode(tsl_text[0], skip_special_tokens=True)
    # return the translated text
    return tsl_text


def translate_beam_search(model, tokenizer, text):
    """
    Uses beam search and gives multiple possible translations
    :param model:
    :param tokenizer:
    :param text:
    :return:
    """
    # encode the text
    encoded_text = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    # translate the text
    tsl_text = model.generate(encoded_text, num_beams=5, num_return_sequences=5, early_stopping=True)

    return [tokenizer.decode(t, skip_special_tokens=True) for t in tsl_text]


text_wiki = "Korean dramas are popular worldwide, especially in Asia, partially due to the spread of " \
            "Korean popular culture (the 'Korean Wave'), and their widespread availability via streaming" \
            " services which often offer subtitles in multiple languages."

# basic_translation("en", "de", text_wiki)
# Try checking this on Google Translate :- https://www.google.com/search?q=english+to+german


# Now let's try to use our model and tokenizer
model, tokenizer = get_translation_model_and_tokenizer("en", "zh")
# translated_text = translate_using_model_tokenizer(model, tokenizer, text_wiki)
# print(translated_text)

# Let's try the beam search
translated_text = translate_beam_search(model, tokenizer, text_wiki)
for tsl in translated_text:
    print(tsl)
    print('=' * 50)

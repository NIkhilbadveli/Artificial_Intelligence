"""
There are 2 types of chatbots.
1) One is retrieval-based chatbot that can only output from fixed set of responses. Essentially, we try to classify
the user input into a set of pre-defined classes and choose a response based on the class.
2) The other is a generative chatbot that uses a seq2seq model or encoder-decoder architecture. First the user input
is encoded and then the decoder takes the output of the encoder and produces the output response of the chatbot.

Surprisingly, it looks like the popular digital assistants like Google Assistant, Siri etc., use the first approach.
And the second type is still under research. I guess there's more control in the first type, at least if you consider
business needs.

"""
import json
import random
import time

import nltk
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt')
# Download the wordnet lemmatizer
nltk.download('wordnet')
# Download the stopwords corpus
nltk.download('stopwords')


def clean_corpus(corpus):
    """
    This function cleans the corpus by removing stopwords, punctuations, and lemmatizing the words.
    :param corpus:
    :return:
    """
    # lowering every word in text
    corpus = [doc.lower() for doc in corpus]
    cleaned_corpus = []

    stop_words = stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    # iterating over every text
    for doc in corpus:
        # tokenizing text
        tokens = word_tokenize(doc)
        cleaned_sentence = []
        for token in tokens:
            # removing stopwords, and punctuation
            if token not in stop_words and token.isalpha():
                # applying lemmatization
                cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token))
        cleaned_corpus.append(' '.join(cleaned_sentence))
    return cleaned_corpus


def get_train_data(intents):
    """
    This function creates the x (all pattern sentences in all tags) and y (all tags) data used for training the model.
    So, for every pattern sentence, we have a tag.

    Also, we will clean the corpus and vectorize the x values before returning the data.
    :param intents:
    :return:
    """
    all_patterns = []
    tag_for_pattern = []
    for intent in intents:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            tag_for_pattern.append(intent['tag'])

    # cleaning the corpus
    cleaned_corpus = clean_corpus(all_patterns)
    x = vectorizer.fit_transform(cleaned_corpus).toarray()
    y = encoder.fit_transform(np.array(tag_for_pattern).reshape(-1, 1)).toarray()
    return x, y


def train_dense_model(x, y, epochs=20):
    """
    This function trains a dense model.
    :param x:
    :param y:
    :param epochs:
    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_shape=(x.shape[1],), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x, y, epochs=epochs, batch_size=1)
    return model


def predict_intent_tag(model, message):
    """
    This function predicts the intent tag for the given message.
    :param model:
    :param message:
    :return:
    """
    message = clean_corpus([message])
    x_test = vectorizer.transform(message).toarray()
    y_pred = model.predict(x_test)

    # if probability of all intent is low, classify it as no_answer
    if y_pred.max() < 0.4:
        return 'no_answer'

    prediction = np.zeros_like(y_pred[0])
    prediction[y_pred.argmax()] = 1
    print('Prediction vector', prediction)
    print(encoder.inverse_transform([prediction]))
    tag = encoder.inverse_transform([prediction])[0][0]
    return tag, get_intent(tag)


def perform_action(action_code, intent):
    """
    This function performs the action for the given intent.
    :param action_code:
    :param intent:
    :return:
    """
    # funition to perform an action which is required by intent

    if action_code == 'CHECK_ORDER_STATUS':
        print('\n Checking database \n')
        time.sleep(2)
        order_status = ['in kitchen', 'with delivery executive']
        delivery_time = []
        return {'intent-tag': intent['next-intent-tag'][0],
                'order_status': random.choice(order_status),
                'delivery_time': random.randint(10, 30)}

    elif action_code == 'ORDER_CANCEL_CONFIRMATION':
        ch = input('BOT: Do you want to continue (Y/n) ?')
        if ch == 'y' or ch == 'Y':
            choice = 0
        else:
            choice = 1
        return {'intent-tag': intent['next-intent-tag'][choice]}

    elif action_code == 'ADD_DELIVERY_INSTRUCTIONS':
        instructions = input('Your Instructions: ')
        return {'intent-tag': intent['next-intent-tag'][0]}


def get_intent(tag):
    """
    This function returns the intent for the given tag.
    :param tag:
    :return:
    """
    # to return complete intent from intent tag
    for intent in json_data['intents']:
        if intent['tag'] == tag:
            return intent


# Read and print first row of data
with open('intent.json', 'r') as f:
    json_data = json.load(f)
    # The structure of this file is "intents" (list) -> "tag", "patterns" (list), "responses" (list).
    # Some of the intents might have "action" and other keys.
    print(json_data['intents'][0])

# vectorizing the corpus
vectorizer = TfidfVectorizer()
# one hot encoding the tags
encoder = OneHotEncoder()

# the above vectorizer and encoder are used inside the train_dense_model function
x_train, y_train = get_train_data(json_data['intents'])
print('Shapes of x_train and y_train: ', x_train.shape, y_train.shape)
model = train_dense_model(x_train, y_train)
print('Model Trained')
print('\n')
print('Now the bot will start talking to you.\n')

# Test the input sequence :- "hi" -> "order status" -> "no"
while True:
    # get message from user
    input_text = input('You: ')
    tag, intent = predict_intent_tag(model, input_text)
    # generate random response from intent
    response = random.choice(intent['responses'])
    print('Bot: ', response)

    # check if there's a need to perform some action
    if 'action' in intent.keys():
        action_code = intent['action']
        # perform action
        data = perform_action(action_code, intent)
        # get follow up intent after performing action
        followup_intent = get_intent(data['intent-tag'])
        # generate random response from follow up intent
        response = random.choice(followup_intent['responses'])

        # print randomly selected response
        if len(data.keys()) > 1:
            print('Bot: ', response.format(**data))
        else:
            print('Bot: ', response)

    # break loop if intent was goodbye
    if tag == 'goodbye':
        break

from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

import numpy as np 
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import openai

# load your data
data = load_dataset("rotten_tomatoes")
print (data['train'][0,-1])
client = openai.OpenAI(api_key="")

def evaluate_perfomance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names = ["Negative Review", "Positive Review"] 
    )
    print (performance)


def pre_trained_model():
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    pipe = pipeline(
        model = model_path,
        tokenizer = model_path,
        return_all_scores = True,
        device = 'cpu'
    )

    # Run inference
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data['test'], 'text')), total = len(data['test'])):
        negative_scores = output[0]['score']
        positive_scores = output[2]['score']

        assignment = np.argmax([negative_scores, positive_scores])
        y_pred.append(assignment)

    return y_pred


def create_embeddings():

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Convert text to embeddings
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    train_embeddings.shape

    # Train a logistic regression on our train embeddings
    clf = LogisticRegression(random_state=42)
    clf.fit(train_embeddings, data["train"]["label"])

    # Predict previously unseen instances
    y_pred = clf.predict(test_embeddings)
    return y_pred

def zero_shot_learning():

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Create embeddings for our labels
    label_embeddings = model.encode(["A negative review",  "A positive review"])

    # Convert text to embeddings
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    from sklearn.metrics.pairwise import cosine_similarity

    # Find the best matching label for each document
    sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)

    return y_pred


def generative_models():

    # Load our model
    pipe = pipeline(
        "text2text-generation", 
        model="google/flan-t5-small", 
        device="cuda:0"
    )

    # Prepare our data
    prompt = "Is the following sentence positive or negative? "
    data = data.map(lambda example: {"t5": prompt + example['text']})
    data

    # Run inference
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
        text = output[0]["generated_text"]
        y_pred.append(0 if text == "negative" else 1)

    return y_pred

# ChatGPT for classifcation
def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """Generate an output based on a prompt and an input document."""
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
            },
        {
            "role": "user",
            "content":   prompt.replace("[DOCUMENT]", document)
            }
    ]
    chat_completion = client.chat.completions.create(
      messages=messages,
      model=model,
      temperature=0
    )
    return chat_completion.choices[0].message.content


def chatGPT():
    # Define a prompt template as a base
    prompt = """Predict whether the following document is a positive or negative movie review:

    [DOCUMENT]

    If it is positive return 1 and if it is negative return 0. Do not give any other answers.
    """

    # Predict the target using GPT
    document = "unpretentious , charming , quirky , original"
    doc = "digusting material no reason to watch"
    print ( chatgpt_generation(prompt, doc))

    # You can skip this if you want to save your (free) credits
    predictions = [
        chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])
    ]

    y_pred = [int(pred) for pred in predictions]
    return y_pred

def deepSeek():
    return 0

methodology = "zero_shot_learning"

if methodology == "pre_trained_models":
    y_pred = pre_trained_model()
elif methodology == "embeddings":   
    y_pred = create_embeddings()
elif methodology == "zero_shot_learning":
    y_pred = zero_shot_learning()
elif methodology == "chatGPT":
    y_pred = chatGPT()

evaluate_perfomance(data["test"]["label"], y_pred)

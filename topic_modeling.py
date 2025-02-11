# Load data from Hugging Face
from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
from bertopic import BERTopic

import matplotlib.pyplot as plt
from copy import deepcopy

from transformers import pipeline
from bertopic.representation import TextGeneration
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired
import openai
from bertopic.representation import OpenAI

# Create an embedding for each abstract
def embedding_extraction():

    embedding_model = SentenceTransformer("thenlper/gte-small")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    return embedding_model, embeddings

# We reduce the input embeddings from 384 dimensions to 5 dimensions
def umap_reduction(embeddings, n_components, min_dist, metric, random_state):

    umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings, umap_model

# We fit the model and extract the clusters
def hdbscan_clustering(reduced_embeddings):
    hdbscan_model = HDBSCAN(
        min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
    ).fit(reduced_embeddings)
    clusters = hdbscan_model.labels_
    return clusters, hdbscan_model

def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    
    return df

# How many clusters did we generate?
# embedding_model, embeddings = embedding_extraction()
# np.save('embedding_model.npy', embedding_model)
# np.save('embeddings.npy', embeddings)

reduced_embeddings = np.load('embeddings.npy')

# reduced_embeddings, umap_model = umap_reduction(embeddings, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# clusters, hdbscan_model = hdbscan_clustering(reduced_embeddings)
# len(set(clusters))

# # Print first three documents in cluster 0
# cluster = 0
# for index in np.where(clusters==cluster)[0][:3]:
#     print(abstracts[index][:300] + "... \n")

# # Reduce 384-dimensional embeddings to two dimensions for easier visualization
# reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)

# Create dataframe
df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
to_plot = df.loc[df.cluster != "-1", :]
outliers = df.loc[df.cluster == "-1", :]

pdb.set_trace()
# Plot outliers and non-outliers separately
fig = plt.figure(figsize=(3, 6))
plt.scatter(outliers.x, outliers.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    to_plot.x, to_plot.y, c=clusters_df.cluster.astype(int),
    alpha=0.6, s=2, cmap="tab20b"
)
plt.axis("off")
fig.savefig('clusters.png', dpi=fig.dpi)

# Train our model with our previously defined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

topic_model.get_topic_info()
topic_model.get_topic(0)
topic_model.find_topics("topic modeling")
topic_model.get_topic(22)
topic_model.topics_[titles.index("BERTopic: Neural topic modeling with a class-based TF-IDF procedure")]

# Visualize topics and documents
fig = topic_model.visualize_documents(
    titles, 
    reduced_embeddings=reduced_embeddings, 
    width=1200, 
    hide_annotations=True
)
fig.savefig('topic_modelling.png', dpi=fig.dpi)


# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))

# Visualize barchart with ranked keywords
topic_model.visualize_barchart()
# Visualize relationships between topics
topic_model.visualize_heatmap(n_clusters=30)
# Visualize the potential hierarchical structure of topics
topic_model.visualize_hierarchy()
# Save original representations
original_topics = deepcopy(topic_model.topic_representations_)

## Update our topic representations using KeyBERTInspired
# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)

# Update our topic representations to MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)

prompt = """I have a topic that contains the following documents: 
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""

# Update our topic representations using Flan-T5
generator = pipeline("text2text-generation", model="google/flan-t5-small")
representation_model = TextGeneration(
    generator, prompt=prompt, doc_length=50, tokenizer="whitespace"
)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)

prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <short topic label>
"""

# Update our topic representations using GPT-3.5
client = openai.OpenAI(api_key="")
representation_model = OpenAI(
    client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt
)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)

# Visualize topics and documents
fig = topic_model.visualize_document_datamap(
    titles,
    topics=list(range(20)),
    reduced_embeddings=reduced_embeddings,
    width=1200,
    label_font_size=11,
    label_wrap_width=20,
    use_medoids=True,
)


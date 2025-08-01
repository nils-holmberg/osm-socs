# -*- coding: utf-8 -*-
"""embedding_vector_similarity.ipynb"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets

# --- 3D plot of synthetic vectors ---
word_vectors = {
    'king': np.array([0.8, 0.6, 0.9]),
    'queen': np.array([0.7, 0.7, 0.9]),
    'man': np.array([0.9, 0.5, 0.6]),
    'woman': np.array([0.6, 0.8, 0.6]),
    'apple': np.array([0.1, 0.2, 0.9])
}

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for word, vec in word_vectors.items():
    ax.scatter(*vec, label=word)
    ax.text(*vec, word, size=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of Word Embeddings")
plt.legend()
plt.savefig("plot_word_vectors.png")
plt.close()

# --- Cosine similarity matrix ---
words = list(word_vectors.keys())
vectors = np.array([word_vectors[word] for word in words])
similarities = cosine_similarity(vectors)
sim_df = pd.DataFrame(similarities, index=words, columns=words)
sim_df.to_csv("cosine_similarity_matrix.csv")

# --- BERT word embedding PCA plot ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(word):
    tokens = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[0, 1].numpy()

words = ["bank", "money", "river", "finance", "stream"]
embeddings = [get_embedding(word) for word in words]
reduced = PCA(n_components=3).fit_transform(embeddings)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for word, vec in zip(words, reduced):
    ax.scatter(*vec, label=word)
    ax.text(*vec, word, size=10)
ax.set_title("3D PCA of BERT Word Embeddings")
plt.legend()
plt.savefig("plot_bert_words_pca.png")
plt.close()

# --- IMDb review embeddings ---
dataset = load_dataset("imdb", split="train").shuffle(seed=42)
pos = dataset.filter(lambda x: x['label'] == 1).select(range(25))
neg = dataset.filter(lambda x: x['label'] == 0).select(range(25))
balanced = concatenate_datasets([pos, neg]).shuffle(seed=42)
texts = [x['text'][:300] for x in balanced]
labels = [x['label'] for x in balanced]

def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

embeddings = [get_cls_embedding(text) for text in texts]
reduced_embeddings = PCA(n_components=3).fit_transform(embeddings)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for i, vec in enumerate(reduced_embeddings):
    color = 'blue' if labels[i] == 0 else 'red'
    ax.scatter(*vec, c=color, alpha=0.6)
ax.set_title("3D PCA of BERT Review Embeddings (Blue=Neg, Red=Pos)")
plt.savefig("plot_imdb_pca.png")
plt.close()

# --- Classification ---
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("IMDb BERT classification accuracy:", accuracy_score(y_test, preds))

# --- Genre similarity inference ---
genre_labels = ["science fiction", "drama", "thriller", "comedy", "documentary"]
genre_embeddings = {label: get_cls_embedding(label) for label in genre_labels}

examples = [
    {"title": "Interstellar", "text": "A brilliant depiction of space travel, black holes, and human survival.", "genre": "science fiction"},
    {"title": "The Godfather", "text": "An intense family drama centered around the mafia underworld.", "genre": "drama"},
    {"title": "Inception", "text": "A mind-bending thriller about dreams within dreams.", "genre": "thriller"},
    {"title": "Superbad", "text": "A hilarious coming-of-age story full of awkward teen moments.", "genre": "comedy"},
    {"title": "The Social Dilemma", "text": "An eye-opening documentary about social media's effects on society.", "genre": "documentary"},
]

review_embeddings = [get_cls_embedding(example['text']) for example in examples]

inferred_genres = []
for review_vec in review_embeddings:
    sims = {genre: cosine_similarity([review_vec], [vec])[0][0] for genre, vec in genre_embeddings.items()}
    best_match = max(sims, key=sims.get)
    inferred_genres.append(best_match)

final_df = pd.DataFrame({
    "Title": [ex['title'] for ex in examples],
    "Actual Genre": [ex['genre'] for ex in examples],
    "Inferred Genre": inferred_genres
})

final_df.to_csv("genre_prediction_results.csv", index=False)
print(final_df)

import csv
import os

import jsonlines
import openai
import numpy as np
import time
from tqdm import tqdm
from utils import riddle_list


def request_embedding(input, model='text-embedding-ada-002'):
    response = openai.Embedding.create(
        input=input,
        model=model,
    )
    with jsonlines.open('../data/embedding.jsonl', mode='a') as w:
        w.write([input, response])


def embedding_data():
    data = [r['question'] for r in riddle_list]
    with jsonlines.open('../data/embedding.jsonl') as r:
        response_data = [res for res in r]
    data = data[len(response_data):]
    for d in tqdm(data):
        request_embedding(d)
        time.sleep(0.5)


def cal_similarity():
    with jsonlines.open('../data/embedding.jsonl') as r:
        embeddings = [res[1] for res in r]
    N = len(riddle_list)
    assert N == len(embeddings)
    similarity_matrix = np.zeros((N, N))
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            e1 = embeddings[i]
            e2 = embeddings[j]
            similarity_matrix[i, j] = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            similarity_matrix[j, i] = similarity_matrix[i, j]
    np.save('../data/similarity.npy', similarity_matrix)


def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embedding_data()
    with jsonlines.open('../data/embedding.jsonl') as r:
        data = [res for res in r]
    for i in range(len(data)):
        data[i] = [data[i][0], data[i][1]['data'][0]['embedding']]
    with jsonlines.open('../data/embedding.jsonl', mode='w') as w:
        w.write_all(data)
    cal_similarity()


if __name__ == "__main__":
    main()

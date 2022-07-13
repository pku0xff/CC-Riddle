from sentence_transformers import SentenceTransformer, util
from create_train_data import read_expl, read_ids
import time
import json
import re

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'output/training_bert-base-chinese'
bi_encoder = SentenceTransformer(model_name)
top_k = 10  # Number of passages we want to retrieve with the bi-encoder

# passages: [title, text]
# candidate answers: [character, explanation]
def read_candidates():
    characters = []
    lines = open('../data/all_dict.txt').read().strip().split('\n')
    for line in lines:
        line = line.split(',')
        characters.append(line[0])
    return characters

passages = []
characters = read_candidates()
# explanation = read_expl()
explanation = read_ids()
for ch in characters:
    try:
        # expl = ch + ' [SEP] ' + explanation[ch]
        expl = explanation[ch]
    except:
        # expl = ch + ' [SEP] '
        expl = ch
    passages.append([ch, expl])

print("Passages:", len(passages))


corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

flag = True

generated_list = []
lines = open('../data/generated_list.txt', encoding='utf-8').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    generated_list.append((line[0], line[1]))

lines = open('../data/test_set.txt').read().strip().split('\n')
total = 0
correct = 0
MRR_3 = 0.0
MRR_5 = 0.0
MRR_10 = 0.0
start_time = time.time()
# f = open('bert-base-chinese-ids.txt', 'w', encoding='utf-8')
cnt_generated_solved = 0.0
cnt_generated = 0
cnt_all_solved = 0.0
for line in lines:
    line = line.split(',')
    ch = line[0]
    riddles = line[1:]
    total += len(riddles)
    for query in riddles:

        question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query
        # print("Input question:", query, ch)
        # f.write(f"Input question:, {query}, {ch}\n")
        if (ch, query) in generated_list:
            cnt_generated += 1

        for i in range(5):
            hit = hits[i]
            # print(i + 1, "\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']][0]))
            # f.write(str(i + 1))
            # f.write("\t{:.3f}\t{}\n".format(hit['score'], passages[hit['corpus_id']][0]))
            if passages[hit['corpus_id']][0] == ch:
                cnt_all_solved += 1
                if (passages[hit['corpus_id']][0], query) in generated_list:
                    cnt_generated_solved += 1
                MRR_10 += (10 - i) / 10
                if i == 1:
                    # print("Input question:", query, ch)
                    # print(i + 1, "\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']][0]))
                    correct += 1
                if i < 3:
                    MRR_3 += (3 - i) / 3
                if i < 5:
                    MRR_5 += (5 - i) / 5
# f.close()
print("correct: ", correct / total)
print("MRR for top 3: ", MRR_3 / total)
print("MRR for top 5: ", MRR_5 / total)
print("MRR for top 10: ", MRR_10 / total)
end_time = time.time()
print("Finished after {:.3f} seconds.".format(end_time - start_time))

print(cnt_generated_solved, cnt_all_solved, cnt_generated_solved / cnt_all_solved)
print(cnt_generated, total, cnt_generated / total)

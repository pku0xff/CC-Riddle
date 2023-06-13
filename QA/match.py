"""
modified from:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py
"""
import argparse
import csv
import json
import logging
import math
import random
import re
import jsonlines
from collections import defaultdict
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
from utils import riddle_list


def main():
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--mode', type=str, default='glyph')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    stage = args.stage
    mode = args.mode
    model_name = args.model
    train_batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    num_epochs = args.num_epochs
    save_path = args.save_path

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    info = defaultdict(str)
    if mode == 'glyph':
        with open('../data/src_data/ids.txt') as f:
            lines = f.readlines()
        lines = lines[2:]
        for l in lines:
            l = l.split('\t')
            info[l[1].strip()] = l[2].strip()
    elif mode == 'meaning':
        with open('../data/src_data/word.json', 'r', encoding='utf-8') as f:
            word_dict_list = json.load(f)
            for word_dict in word_dict_list:
                ch = word_dict["word"]
                expl = word_dict["explanation"]
                expl = re.sub('\n \n', '\n\n', expl)
                expl = '。'.join(expl.split('\n\n')[:5])
                expl = re.sub(r',', '，', expl)
                expl = re.sub(r'\n', '', expl)
                info[ch] = expl
    else:
        raise ValueError(f'Unknown mode: {mode}')

    if stage == 'train':
        # Here we define our SentenceTransformer model
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Read the file and create the training dataset
        logging.info("Read train dataset")

        def add_to_samples(sent1, sent2, label):
            if sent1 not in train_data:
                train_data[sent1] = {'contradiction': set(), 'entailment': set()}
            train_data[sent1][label].add(sent2)

        train_data = {}

        def construct_train_data(train_set):
            for entry in train_set:
                question = entry['question']
                answer = entry['answer']
                add_to_samples(question, f'{answer}，{info[answer]}', 'entailment')
                add_to_samples(f'{answer}，{info[answer]}', question, 'entailment')
                # Add negative samples
                negative_a = random.choice(train_set)
                while negative_a['answer'] == answer:
                    negative_a = random.choice(train_set)
                add_to_samples(question, f'{negative_a["answer"]}，{info[negative_a["answer"]]}', 'contradiction')
                add_to_samples(f'{negative_a["answer"]}，{info[negative_a["answer"]]}', question, 'contradiction')
                negative_q = random.choice(train_set)
                while negative_q['answer'] == answer:
                    negative_q = random.choice(train_set)
                add_to_samples(negative_q['question'], f'{answer}，{info[answer]}', 'contradiction')
                add_to_samples(f'{answer}，{info[answer]}', negative_q['question'], 'contradiction')

        train_set = [e for e in riddle_list if e['split'] == 'train']
        construct_train_data(train_set)

        train_samples = []
        for sent1, others in train_data.items():
            if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
                train_samples.append(InputExample(
                    texts=[sent1, random.choice(list(others['entailment'])),
                           random.choice(list(others['contradiction']))]))
                train_samples.append(InputExample(
                    texts=[random.choice(list(others['entailment'])), sent1,
                           random.choice(list(others['contradiction']))]))

        logging.info("Train samples: {}".format(len(train_samples)))

        # Special data loader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Read the file and create the validation dataset
        valid_set = [e for e in riddle_list if e['split'] == 'valid']
        valid_samples = []
        for entry in valid_set:
            question = entry['question']
            answer = entry['answer']
            valid_samples.append(InputExample(texts=[question, f'{answer}，{info[answer]}'], label=1))
            # Add negative samples
            negative_a = random.choice(valid_set)
            while negative_a['answer'] == answer:
                negative_a = random.choice(valid_set)
            valid_samples.append(
                InputExample(texts=[question, f'{negative_a["answer"]}，{info[negative_a["answer"]]}'], label=0))
            negative_q = random.choice(valid_set)
            while negative_q['answer'] == answer:
                negative_q = random.choice(valid_set)
            valid_samples.append(InputExample(texts=[negative_q['question'], f'{answer}，{info[answer]}'], label=0))

        valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, batch_size=train_batch_size,
                                                                           name='CC-Riddle_valid')

        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=valid_evaluator,
                  epochs=num_epochs,
                  evaluation_steps=int(len(train_dataloader) * 0.1),
                  warmup_steps=warmup_steps,
                  output_path=save_path,
                  use_amp=False  # Set to True, if your GPU supports FP16 operations
                  )

    # Test the model
    candidates = []
    for entry in riddle_list:
        answer = entry['answer']
        candidates.append(f'{answer}，{info[answer]}')
    candidates = list(set(candidates))
    print("Candidates: {}".format(len(candidates)))

    # Load test dataset
    test_set = [e for e in riddle_list if e['split'] == 'test']

    # Load the stored model and evaluate its performance
    model = SentenceTransformer(save_path)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True, show_progress_bar=True)
    results = []
    acc = 0
    for entry in tqdm(test_set):
        question = entry['question']
        question_embedding = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        hits = util.semantic_search(question_embedding, candidate_embeddings, top_k=5)
        hits = hits[0]
        hit_characters = [candidates[hit['corpus_id']][0] for hit in hits]
        entry['pred'] = hit_characters
        results.append(entry)
        if hit_characters[0] == entry["answer"]:
            acc += 1
    print("Accuracy: {}".format(acc / len(test_set)))
    with jsonlines.open(f'{save_path}/results.jsonl', 'w') as writer:
        writer.write_all(results)


if __name__ == '__main__':
    main()

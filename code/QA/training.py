"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the CC-Riddle dataset
with MultipleNegativesRankingLoss.
Usage:
python training.py
OR
python training.py pretrained_transformer_model_name
"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'nghuyong/ernie-1.0'
train_batch_size = 16  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 128
num_epochs = 1

# Save path of the model
model_save_path = 'output/training_ids_' + model_name.replace("/", "-")

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read the file and create the training dataset
logging.info("Read train dataset")


def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)


train_data = {}
lines = open('train_set_ids.csv', encoding='utf-8').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    sent1 = line[0]
    sent2 = line[1]
    label = line[2]
    add_to_samples(sent1, sent2, label)
    add_to_samples(sent2, sent1, label)

train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        train_samples.append(InputExample(
            texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
        train_samples.append(InputExample(
            texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

logging.info("Train samples: {}".format(len(train_samples)))

# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Read development set
logging.info("Read dev dataset")
dev_samples = []
lines = open('dev_set_ids.csv', encoding='utf-8').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    sent1 = line[0]
    sent2 = line[1]
    label = line[2]
    dev_samples.append(InputExample(texts=[sent1, sent2], label=1 if label == 'entailment' else 0))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='CC-Riddle-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader) * 0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False  # Set to True, if your GPU supports FP16 operations
          )

# create train data for QA model
import json
import random
import csv
import re


def read_expl():
    explanation_dict = {}
    with open('../data/src_data/word.json', 'r', encoding='utf-8') as f:
        word_dict_list = json.load(f)
        for word_dict in word_dict_list:
            ch = word_dict["word"]
            expl = word_dict["explanation"]
            expl = re.sub(r',', '，', expl)
            expl = re.sub(r'\n', '', expl)
            expl = re.sub(r'\n', '', expl)
            explanation_dict[ch] = expl
    return explanation_dict


def read_ids():
    print("Reading ids dataset...")
    ids = {}
    pattern1 = re.compile(r'\[(.*)]')
    pattern2 = re.compile(r'[\[\]]')
    lines = open('../data/src_data/ids.txt', encoding='utf-8').read().strip().split('\n')
    lines = lines[2:]
    for line in lines:
        line = line.split()
        word = line[1]
        decomp = pattern1.sub('', line[2])
        decomp = pattern2.sub('', decomp)
        ids[word] = decomp
    # print(ids)
    return ids


ids_dict = read_ids()
explanation_dict = read_expl()

# train set
train_characters = []
train_questions = []
lines = open('../data/train_set.txt').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    train_characters.append(line[0])
    train_questions += line[1:]

train_set = []
lines = open('../data/train_set.txt').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    raw_ch = line[0]
    try:
        ids = ids_dict[raw_ch]
    except:
        ids = ''
    try:
        expl = explanation_dict[raw_ch]
    except:
        expl = ''
    ch = raw_ch + ' [SEP] ' + ids + ' [SEP] ' + expl
    riddles = line[1:]
    for riddle in riddles:
        train_set.append([ch, riddle, 'entailment'])

        wrong_answer = random.choice(train_characters)
        while wrong_answer == raw_ch:
            wrong_answer = random.choice(train_characters)
        w_expl = wrong_answer
        try:
            w_expl = explanation_dict[wrong_answer]
        except:
            pass
        train_set.append([w_expl, riddle, 'contradiction'])

        wrong_question = random.choice(train_questions)
        while wrong_question in riddles:
            wrong_question = random.choice(train_questions)
        train_set.append([ch, wrong_question, 'contradiction'])

with open('train_set_ids.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(train_set)

# dev set
dev_characters = []
dev_questions = []
lines = open('../data/valid_set.txt').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    dev_characters.append(line[0])
    dev_questions += line[1:]

dev_set = []
lines = open('../data/valid_set.txt').read().strip().split('\n')
for line in lines:
    line = line.split(',')
    raw_ch = line[0]
    try:
        ids = ids_dict[raw_ch]
    except:
        ids = ''
    try:
        expl = explanation_dict[raw_ch]
    except:
        expl = ''
    ch = raw_ch + ' [SEP] ' + ids + ' [SEP] ' + expl
    riddles = line[1:]
    for riddle in riddles:
        dev_set.append([ch, riddle, 'entailment'])

        wrong_answer = random.choice(dev_characters)
        while wrong_answer == raw_ch:
            wrong_answer = random.choice(dev_characters)
        w_expl = wrong_answer
        try:
            w_expl = explanation_dict[wrong_answer]
        except:
            pass
        dev_set.append([w_expl, riddle, 'contradiction'])

        wrong_question = random.choice(dev_questions)
        while wrong_question in riddles:
            wrong_question = random.choice(dev_questions)
        dev_set.append([ch, wrong_question, 'contradiction'])

with open('dev_set_ids.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(dev_set)

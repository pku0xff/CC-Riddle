import csv
import jsonlines
import json
import numpy as np
import openai
import random
import re
import string
import tiktoken
import time
from collections import defaultdict
from tqdm import tqdm

with jsonlines.open('../data/riddle.jsonl') as r:
    riddle_list = [i for i in r]

with jsonlines.open('../data/decomposition.jsonl') as r:
    decomp_list = [i for i in r]

random.seed(42)

decomp_dict = {}
for l in decomp_list:
    ch = l['character']
    del l['character']
    decomp_dict[ch] = l


def num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo-0301") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def complete(messages, model, max_tokens, temperature=1):
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def complete_set(test_set, template, save_path, model='gpt-3.5-turbo-0301', max_tokens=128, temperature=1):
    response_file = '.'.join(save_path.split('.')[:-1]) + '_response.jsonl'
    with jsonlines.open(f'{save_path}', 'a') as w1, jsonlines.open(response_file, 'a') as w2:
        for entry in tqdm(test_set):
            msg = template(entry)
            response = complete(msg, model, max_tokens=max_tokens, temperature=temperature)
            w2.write(response)
            pred = response['choices'][0]['message']['content']
            entry['pred'] = pred
            w1.write(entry)
            time.sleep(20)


def create_choices(strategy='random', k=4):
    test_data = [entry for entry in riddle_list if entry['split'] == 'test']

    if strategy == 'top':
        similarity = np.load('../data/similarity.npy')
    candidates = set(entry['answer'] for entry in riddle_list)

    for i in tqdm(range(len(test_data))):
        character = test_data[i]['answer']
        if strategy == 'random':
            choices = random.sample(list(candidates - {character}), k - 1)
            choices.append(character)
            random.shuffle(choices)
            test_data[i]['choices'] = choices
        elif strategy == 'top':
            sim = similarity[i]
            order = np.argsort(-sim)
            choices = [character]
            for j in order:
                if len(choices) == k:
                    break
                if riddle_list[j]['answer'] not in choices:
                    choices.append(riddle_list[j]['answer'])
            random.shuffle(choices)
            test_data[i]['choices'] = choices
        else:
            raise ValueError
    with jsonlines.open(f'../data/test_{strategy}.jsonl', 'w') as w:
        w.write_all(test_data)


# TODO: move this function to generation folder
def extract_alignment():
    pair_count = defaultdict(int)
    train_data = [entry for entry in riddle_list if entry['split'] == 'train']
    punct = re.compile(r'[，。、”“’‘；!！\'"《》,.\-—]')
    for entry in train_data:
        character = entry['answer']
        if character not in decomp_dict:
            continue
        description = entry['question']
        description = punct.sub('', description)
        decomp = set(decomp_dict[character]['once'] +
                     decomp_dict[character]['radical'] +
                     decomp_dict[character]['graphical'])
        for i in range(0, len(description)):
            for j in range(i + 1, len(description)):
                phrase = description[i:j]
                for d in decomp:
                    if d not in phrase:
                        pair_count[(phrase, d)] += 1
    sorted_dict = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = [(k, v) for k, v in sorted_dict if v > 2]
    with open('../data/alignment.csv', 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phrase', 'decomp', 'count'])
        for k, v in sorted_dict:
            writer.writerow([k[0], k[1], v])


def count(dataset):
    print(len(dataset))
    characters = set([e['answer'] for e in dataset])
    print('characters', len(characters))
    human = [e for e in dataset if e['source'] == 'human']
    print('human', len(human))
    human_ch = set([e['answer'] for e in human])
    print('human_ch', len(human_ch))
    gen = [e for e in dataset if e['source'] == 'gen']
    print('gen', len(gen))
    gen_ch = set([e['answer'] for e in gen])
    print('gen_ch', len(gen_ch))


def dataset_info():
    print('all')
    count(riddle_list)
    print('train')
    train_data = [entry for entry in riddle_list if entry['split'] == 'train']
    count(train_data)
    print('valid')
    valid_data = [entry for entry in riddle_list if entry['split'] == 'valid']
    count(valid_data)
    print('test')
    test_data = [entry for entry in riddle_list if entry['split'] == 'test']
    count(test_data)


def report_results(filename):
    print(filename)
    with jsonlines.open(filename) as r:
        data = [e for e in r]
    gen = [e for e in data if e['source'] == 'gen']
    human = [e for e in data if e['source'] == 'human']
    if 'bert' in filename or 'BERT' in filename:
        acc = 0
        acc_gen = 0
        acc_human = 0
        mrr = 0
        mrr_gen = 0
        mrr_human = 0
        for entry in data:
            if entry['answer'] == entry['pred'][0]:
                acc += 1
                if entry['source'] == 'gen':
                    acc_gen += 1
                else:
                    acc_human += 1
                # print(entry['answer'], entry['pred'])
            if entry['answer'] in entry['pred']:
                mrr_entry = 1 / (entry['pred'].index(entry['answer']) + 1)
                mrr += mrr_entry
                if entry['source'] == 'gen':
                    mrr_gen += mrr_entry
                else:
                    mrr_human += mrr_entry
        print(f'acc: {acc / len(data)}')
        print(f'acc_human: {acc_human / len(human)}')
        print(f'acc_gen: {acc_gen / len(gen)}')
        # print(f'mrr_gen: {mrr_gen / len(gen)}')
        # print(f'mrr_human: {mrr_human / len(human)}')
        # print(f'mrr_all: {mrr / len(data)}')
    else:
        acc_gen = []
        acc_human = []
        err_ids = []
        for entry in data:
            if entry['pred'] == entry['answer']:
                if entry['source'] == 'human':
                    acc_human.append(entry)
                else:
                    acc_gen.append(entry)
            else:
                err_ids.append(entry['id'])
        print(f'gen: {len(acc_gen)}/{len(gen)}={len(acc_gen) / len(gen)}')
        print(f'human: {len(acc_human)}/{len(human)}={len(acc_human) / len(human)}')
        print(f'overall: {len(acc_gen) + len(acc_human)}/{len(data)}={(len(acc_gen) + len(acc_human)) / len(data)}')


def report_coverage():
    characters = set([e['answer'] for e in riddle_list])
    gen = set([e['answer'] for e in riddle_list if e['source'] == 'gen'])
    gb = open('../data/gb.txt', 'r', encoding='utf8').read().strip().split('\n')
    gb = set([e.split('\t')[1] for e in gb[1:]])
    tgscc_data = open('../data/tgscc.txt', 'r', encoding='utf8').read().strip().split('\n')
    tgscc = set([e.split('\t')[1] for e in tgscc_data[3:]])
    freq = set([e.split('\t')[1] for e in tgscc_data[3:]][:3500])
    print('gb', len(gb))
    print('gb coverage', f'{len(characters.intersection(gb))} / {len(gb)}',
          len(characters.intersection(gb)) / len(gb))
    print('gen gb coverage', f'{len(gen.intersection(gb))} / {len(gb)}', len(gen.intersection(gb)) / len(gb))
    print('tgscc', len(tgscc))
    print('tgscc coverage', f'{len(characters.intersection(tgscc))} / {len(tgscc)}',
          len(characters.intersection(tgscc)) / len(tgscc))
    print('gen tgscc coverage', f'{len(gen.intersection(tgscc))} / {len(tgscc)}',
          len(gen.intersection(tgscc)) / len(tgscc))
    print('freq', len(freq))
    print('freq coverage', f'{len(characters.intersection(freq))} / {len(freq)}',
          len(characters.intersection(freq)) / len(freq))
    print('gen freq coverage', f'{len(gen.intersection(freq))} / {len(freq)}',
          len(gen.intersection(freq)) / len(freq))


def error_analysis(filelist):
    all_data = {}
    for filename in filelist:
        error = []
        if 'jsonl' in filename:
            with jsonlines.open(filename) as r:
                data = [e for e in r]
            for e in data:
                if 'output' in filename:
                    if e['answer'] not in e['pred']:
                        error.append(e)
                else:
                    if e['pred'] != e['answer']:
                        error.append(e)
        elif 'csv' in filename:
            with jsonlines.open('annotate.jsonl') as r:
                annotate_data = [e for e in r]
            with open(filename) as f:
                reader = csv.reader(f)
                data = [l for l in reader]
            data = data[1:]
            for i, l in enumerate(data):
                if annotate_data[i]['answer'] not in l[1]:
                    tmp = annotate_data[i]
                    tmp['pred'] = l[1]
                    error.append(tmp)
        all_data[filename] = error
    all_fail = set()
    for filename, error in all_data.items():
        if len(all_fail) == 0:
            all_fail = set([e['id'] for e in error])
        else:
            all_fail = all_fail.intersection(set([e['id'] for e in error]))
    all_fail = list(all_fail)
    gen_fail = 0
    human_fail = 0
    for id in all_fail:
        if riddle_list[id]['source'] == 'gen':
            gen_fail += 1
        else:
            human_fail += 1
    print('all fail', len(all_fail))
    print('gen fail', gen_fail)
    print('human fail', human_fail)
    sampled_error = random.sample(all_fail, min(10, len(all_fail)))
    for id in sampled_error:
        print(riddle_list[id])
        for filename, error in all_data.items():
            if 'mc' in filename or 'meaning' in filename or '2' in filename or '3' in filename:
                continue
            print(filename)
            for e in error:
                if e['id'] == id:
                    print(e['pred'])



import re
import json


def add_space(src):
    l = list(src)
    tgt = str(l).replace('[', '').replace(']', '')
    tgt = tgt.replace("'", '').replace(',', '')
    # print(src)
    # print(tgt)
    return tgt


def read_once():
    print("Reading once table...")
    once_dict = {}
    lines = open('src-data/once_table.txt', encoding='utf-8').read().strip().split('\n')
    for line in lines:
        once_dict[line[0]] = line[2:]
    # print(once_dict)
    return once_dict


def read_radical():
    print("Reading radical table...")
    radical_dict = {}
    lines = open('src-data/radical_table.txt', encoding='utf8').read().strip().split('\n')
    for line in lines:
        radical_dict[line[0]] = line[2:]
    # print(radical_dict)
    return radical_dict


def read_pinyin():
    print("Reading pinyin...")
    pinyin_dict = {}
    lines = open('src-data/pinyin_dict.txt', encoding='utf8').read().strip().split('\n')
    for line in lines:
        pinyin_dict[line[0]] = line[2:]
    return pinyin_dict


def read_xhzd(word_set):
    explanation_dict = {}
    print("Reading Xinhua Dictionary...")
    punctuation_pattern1 = re.compile(r'[";,\'\\]')
    punctuation_pattern2 = re.compile(r'[\n]')
    # like {"word": ["explanation1", "explanation2", ...], ...}
    with open('src-data/word.json', 'r', encoding='utf-8') as f:
        word_dict_list = json.load(f)
        # print(info_dict_list)
        for word_dict in word_dict_list:
            word = word_dict["word"]
            if word not in word_set:
                continue
            expl = word_dict["explanation"]
            expl = punctuation_pattern1.sub('', expl)
            expl = punctuation_pattern2.sub(' ', expl)
            if word in explanation_dict.keys():
                explanation_dict[word] += ' ' + expl
            else:
                explanation_dict[word] = expl
    return explanation_dict


def read_riddles():
    print("Reading riddles...")
    # Read the file and split into lines
    lines = open("src-data/riddle_dict.txt", encoding='utf-8').read().strip().split('\n')
    # 此处得到的 line 格式为 ['字 字谜1 字谜2 ...']
    riddle_dict = {}
    for l in lines:
        l = l.split()
        try:
            riddle_dict[l[0]] = l[1:]
        except Exception as err:
            print(err)
    return riddle_dict


def read_ids():
    print("Reading ids dataset...")
    ids = {}
    pattern1 = re.compile(r'\[(.*)]')
    pattern2 = re.compile(r'[\[\]]')
    lines = open('src-data/ids.txt', encoding='utf-8').read().strip().split('\n')
    lines = lines[2:]
    for line in lines:
        line = line.split()
        word = line[1]
        decomp = pattern1.sub('', line[2])
        decomp = pattern2.sub('', decomp)
        ids[word] = decomp
    # print(ids)
    return ids


def read_expl():
    print("Reading explanations...")
    expl_dict = {}
    lines = open('src-data/expl_crawled.txt', encoding='utf-8').read().strip().split('\n')
    for line in lines:
        line = line.split()
        expl = ''
        for l in line[1:]:
            expl += l
        expl_dict[line[0]] = expl
    return expl_dict


def read_expl_2():
    print("Reading explanations...")
    expl_dict = {}
    lines = open('src-data/expl_dict.txt', encoding='utf-8').read().strip().split('\n')
    for line in lines:
        line = line.split()
        expl = ''
        for l in line[1:]:
            expl += l
        expl_dict[line[0]] = expl
    return expl_dict

from __future__ import unicode_literals, print_function, division
from io import open
from utils import (
    add_space,
    read_radical,
    read_riddles,
    read_pinyin,
    read_ids,
    read_expl_2
)
import json
import re
import random


def create_input(ch_set, pinyin_dict, ids_dict, radical_dict, expl_dict):
    print("Creating input...")
    input_dict = {}
    sep = " [SEP] "
    for word in ch_set:
        pinyin = ''
        ids = ''
        radical = ''
        expl = ''
        try:
            pinyin = pinyin_dict[word]
        except Exception as err:
            print(err)
        try:
            ids = ids_dict[word]
        except Exception as err:
            print(err)
        try:
            radical = radical_dict[word]
        except Exception as err:
            print(err)
        try:
            expl = expl_dict[word]
        except Exception as err:
            print(err)
            pass
        input_dict[word] = word + sep + pinyin + sep + add_space(ids) + sep + radical + sep + add_space(expl)
    return input_dict


def write_train(filename, characters, riddle_dict, input_dict):
    print("Saving train set...")
    f = open(filename, 'w', encoding='utf-8')
    for ch in characters:
        riddles = riddle_dict[ch]
        input_info = input_dict[ch]
        for r in riddles:
            r = add_space(r)
            # print('{ "translation": { "en": "' + input_info + '", "ro": "' + r + '" } }')
            f.write('{ "translation": { "en": "' + input_info + '", "ro": "' + r + '" } }\n')
    f.close()


def write_test(filename, characters, input_dict):
    print("Saving test set...")
    f = open(filename, 'w', encoding='utf-8')
    for ch in characters:
        try:
            input_info = input_dict[ch]
            # print(input_info)
            f.write(input_info + '\n')
        except Exception as err:
            print(err)
    f.close()


if __name__ == "__main__":
    radical_dict = read_radical()
    pinyin_dict = read_pinyin()
    ids_dict = read_ids()

    train_ch = open('train_ch.txt', encoding='utf8').read().strip().split('\n')
    validation_ch = open('validation_ch.txt', encoding='utf8').read().strip().split('\n')
    test_ch = open('test_ch.txt', encoding='utf8').read().strip().split('\n')
    print(len(test_ch))
    all_characters = list(radical_dict.keys())
    ch_set = [ch for ch in all_characters if ch not in train_ch]
    expl_dict = read_expl_2()

    input_dict = create_input(test_ch, pinyin_dict, ids_dict, radical_dict, expl_dict)

    riddle_dict = read_riddles()

    write_train(r'all/train.json', train_ch, riddle_dict, input_dict)
    write_test(r'all/validation.src', validation_ch, input_dict)
    write_test(r'all/test.src', test_ch, input_dict)
    # write_test(r'all-long/new_character.src', ch_set, input_dict)
    print("Finished!")

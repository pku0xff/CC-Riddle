from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import math


def remove_space(src):
    src = src.split()
    tgt = ''
    for s in src:
        tgt += s
    return tgt


def add_space(src):
    l = list(src)
    tgt = str(l).replace('[', '').replace(']', '')
    tgt = tgt.replace("'", '').replace(',', '')
    # print(src)
    # print(tgt)
    return tgt


# Save the generated riddles
def save_generation(text_file, score_file, test_character, pred_txt, bleu_topn):
    with open(text_file, 'w', encoding='utf-8') as text_f:
        for i in range(len(test_character)):
            text_f.write(test_character[i])
            for j in pred_txt[i]:
                text_f.write(';' + j)
            text_f.write('\n')
    bleu_top1 = [topn[0].score for topn in bleu_topn]
    # print(bleu_top1)
    avg_top1 = sum(bleu_top1) / len(bleu_top1)
    bleu_topn = [[j.score for j in i] for i in bleu_topn]
    # print(bleu_topn)
    avg_each_ch = [sum(i) / len(i) for i in bleu_topn]
    avg_topn = sum(avg_each_ch) / len(avg_each_ch)
    with open(score_file, 'w', encoding='utf-8') as text_f:
        text_f.write(f"avg bleu of top 1:{avg_top1}\n")
        text_f.write(f"avg bleu of top n:{avg_topn}\n")
        for i in range(len(test_character)):
            text_f.write(f"{test_character[i]},{avg_each_ch[i]}")
            for j in range(len(pred_txt[i])):
                text_f.write(f";{pred_txt[i][j]},{bleu_topn[i][j]}")
            text_f.write('\n')


# Prepare raw data
def read_riddles():
    print("Reading riddles...")
    # Read the file and split into lines
    lines = open("data/src-data/riddle_dict.txt", encoding='utf-8').read().strip().split('\n')
    # 此处得到的 line 格式为 ['字 字谜1 字谜2 ...']
    riddle_dict = {}
    for l in lines:
        l = l.split()
        try:
            riddle_dict[l[0]] = l[1:]
        except Exception as err:
            print(err)
    return riddle_dict


def bleu_1(riddle, references):
    bleu = sacrebleu.corpus_bleu([riddle], [[add_space(ref)] for ref in references])
    return bleu


def bleu_n(riddles, references):
    bleu_list = [bleu_1(rid, references) for rid in riddles]
    return bleu_list


def main():
    args = {
        "text_save_path": 'fnlp-base-no-radical/generate/text-test.txt',
        "score_save_path": 'fnlp-base-no-radical/generate/score-test.txt',
        "input": 'data/no-radical/test.src',
        "model": 'fnlp-base-no-radical/checkpoint-10000',
        "input_max_length": 64,
        "num_beams": 5,
        "num_return_sequences": 5,
        "output_max_length": 15,
        "pad_token_id": 0,
    }
    riddle_dict = read_riddles()

    model_input = open(args["input"], encoding='utf-8').read().strip().split('\n')
    test_character = [m.split()[0] for m in model_input]
    n = len(test_character)
    print(n)

    # Prepare model and model input
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model"])
    tokenizer = AutoTokenizer.from_pretrained(args["model"])

    inputs = tokenizer(model_input, max_length=args["input_max_length"], padding=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    bad_words_ids = [[int(i[1])] for i in input_ids]

    # Use the model line by line
    generate_txts = []
    for i in range(n):
        input = tokenizer([model_input[i]], max_length=args["input_max_length"], padding="max_length", truncation=True,
                          return_tensors='pt')
        input_ids = input['input_ids']
        # print(input_ids)

        pred_ids = model.generate(input_ids=input_ids,
                                  num_beams=args["num_beams"],
                                  max_length=args["output_max_length"],
                                  pad_token_id=args["pad_token_id"],
                                  bad_words_ids=[bad_words_ids[i]],
                                  num_return_sequences=args["num_return_sequences"],
                                  output_attentions=True
                                  )
        '''
        pred_ids = model.generate(input_ids=input_ids,
                                  max_length=args["output_max_length"],
                                  pad_token_id=args["pad_token_id"],
                                  bad_words_ids=[bad_words_ids[i]],
                                  do_sample=True,
                                  top_k=50,
                                  top_p=0.9,
                                  num_return_sequences=args["num_return_sequences"],
                                  )
        '''
        this_character = []
        for p in pred_ids:
            generate_txt = tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            this_character.append(generate_txt)
        # print(model_input[i][0], this_character)
        generate_txts.append(this_character)

    bleu_topn = []
    for i in range(len(test_character)):
        ch = test_character[i]
        riddles = generate_txts[i]
        references = []
        try:
            references = riddle_dict[ch]
        except:
            pass
        bleu_topn.append(bleu_n(riddles, references))

    # save generated riddles and bleu scores
    save_generation(args["text_save_path"], args["score_save_path"], test_character, generate_txts, bleu_topn)


if __name__ == "__main__":
    main()

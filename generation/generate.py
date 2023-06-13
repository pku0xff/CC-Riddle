from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import csv
import sacrebleu


# Prepare raw data
def read_riddles():
    print("Reading riddles...")
    # Read the source riddle file
    lines = open("data/src-data/riddle_dict.txt", encoding='utf-8').read().strip().split('\n')
    riddle_dict = {}
    for l in lines:
        l = l.split()
        try:
            riddle_dict[l[0]] = l[1:]
        except Exception as err:
            print(err)
    return riddle_dict


def bleu_1(riddle, references):
    bleu = sacrebleu.corpus_bleu([riddle], [references])
    return bleu


def bleu_n(riddles, references):
    bleu_list = [bleu_1(rid, references) for rid in riddles]
    return bleu_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='fnlp-base-no-radical/checkpoint-10000')
    parser.add_argument("--save_path", type=str, default='no-radical/generate/text-test.txt')
    parser.add_argument("--input_file", type=str, default='data/no-radical/test.src')
    parser.add_argument("--input_max_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--output_max_length", type=int, default=15)
    parser.add_argument("--pad_token_id", type=int, default=0)
    args = parser.parse_args()

    riddle_dict = read_riddles()

    model_input = open(args.input_file, encoding='utf-8').read().strip().split('\n')
    test_characters = [m.split()[0] for m in model_input]
    n = len(test_characters)
    print(n)

    # Prepare model and model input
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    inputs = tokenizer(model_input, max_length=args.input_max_length, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    bad_words_ids = [[int(i[1])] for i in input_ids]

    # Generate riddles
    gen_riddles = []
    for i in range(n):
        input = tokenizer([model_input[i]], max_length=args.input_max_length, padding="max_length", truncation=True,
                          return_tensors='pt')
        input_ids = input['input_ids']

        pred_ids = model.generate(
            input_ids=input_ids,
            num_beams=args.num_beams,
            max_length=args.output_max_length,
            pad_token_id=args.pad_token_id,
            bad_words_ids=[bad_words_ids[i]],
            num_return_sequences=args.num_return_sequences,
            output_attentions=True
        )
        preds = []
        for p in pred_ids:
            pred = tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds.append(pred)
        gen_riddles.append(preds)

    # Calculate BLEU score
    bleu_topn = []
    for i in range(len(test_characters)):
        ch = test_characters[i]
        riddles = gen_riddles[i]
        references = []
        try:
            references = riddle_dict[ch]
        except:
            pass
        bleu_topn.append(bleu_n(riddles, references))

    bleu_top1 = [topn[0].score for topn in bleu_topn]
    avg_top1 = sum(bleu_top1) / len(bleu_top1)
    bleu_topn = [[j.score for j in i] for i in bleu_topn]
    avg_each_ch = [sum(i) / len(i) for i in bleu_topn]
    avg_topn = sum(avg_each_ch) / len(avg_each_ch)
    print(f"avg bleu of top 1 results:{avg_top1}")
    print(f"avg bleu of top n results:{avg_topn}")

    # Save generated riddles
    with open(args.save_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['character', 'top1', 'top2', 'top3', 'top4', 'top5'])
        for i in range(len(test_characters)):
            writer.writerow([test_characters[i]] + [j[0] for j in test_characters[i]])


if __name__ == "__main__":
    main()

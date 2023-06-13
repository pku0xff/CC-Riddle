'''
Generative QA and Multiple-choice QA
'''
import argparse
import jsonlines
import openai
import os
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils import riddle_list, decomp_dict
from utils import complete, complete_set, num_tokens_from_string


def qa_template_chatgpt(entry):
    return [
        {"role": "user", "content": f"{entry['question']}（打一字）"
         },
    ]


def qa_template_chatglm(entry):
    return f"{entry['question']}（打一字）"


def mc_template_chatgpt(entry):
    return [
        {
            "role": "user",
            "content": f"{entry['question']}（打一字）\n"
                       f"选项：{'，'.join(entry['choices'])}\n"
                       f"请选出正确选项。"
        }
    ]


def mc_template_chatglm(entry):
    return f"{entry['question']}（打一字）\n" \
           f"选项：{'，'.join(entry['choices'])}\n" \
           f"请选出正确选项。"


def calculate_accuracy(results):
    count_correct = 0
    correct_human = 0
    correct_gen = 0
    wrong_human = 0
    wrong_gen = 0
    for entry in results:
        if entry['pred'] == entry['answer']:
            count_correct += 1
            if entry['source'] == 'human':
                correct_human += 1
            else:
                correct_gen += 1
        else:
            if entry['source'] == 'human':
                wrong_human += 1
            else:
                wrong_gen += 1
    print(f'Accuracy: {count_correct}/{len(results)}')
    print(f'Accuracy (human): {correct_human}/{correct_human + wrong_human}')
    print(f'Accuracy (gen): {correct_gen}/{correct_gen + wrong_gen}')


def test_chatglm(mode, strategy, model_path):
    # test chatglm on the test set
    # run on server
    if mode == 'qa':
        save_path = f'./results/chatglm_qa.jsonl'
        template = qa_template_chatglm
    elif mode == 'mc':
        save_path = f'./results/chatglm_mc_{strategy}.jsonl'
        template = mc_template_chatglm
    else:
        raise ValueError
    if mode == 'qa':
        test_data = [entry for entry in riddle_list if entry['split'] == 'test']
    elif mode == 'mc':
        with jsonlines.open(f'../data/test_{strategy}.jsonl') as r:
            test_data = [entry for entry in r]
    else:
        raise ValueError
    with jsonlines.open(save_path, 'a') as w:
        pass
    with jsonlines.open(save_path) as r:
        tested = [entry['id'] for entry in r]
    test_data = [entry for entry in test_data if entry['id'] not in tested]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    with jsonlines.open(save_path, 'a') as w:
        for entry in tqdm(test_data):
            inp = template(entry)
            outp, _ = model.chat(tokenizer, inp, history=[])
            entry['pred'] = outp
            w.write(entry)


def postprocess_chatglm(mode, strategy):
    if mode == 'qa':
        save_path = f'./results/chatglm_qa.jsonl'
        strategy = ''
    elif mode == 'mc':
        save_path = f'./results/chatglm_mc_{strategy}.jsonl'
    else:
        raise ValueError
    with jsonlines.open(save_path) as r:
        results = [entry for entry in r]
    processed_results = []
    for entry in tqdm(results):
        answer = ''
        pred = entry['pred'].split('\n')[0]
        if mode == 'qa':
            pred = re.sub(r'\(.*?\)', '', pred)
            pred = re.sub(r'。', '', pred)
            if len(pred) == 1:
                answer = pred
            else:
                try:
                    answer = re.findall(r'“(.*?)”', pred)[0]
                except:
                    answer = ''
        elif mode == 'mc':
            if '「' in pred:
                answer = re.findall(r'「(.*?)」', pred)[0]
            elif '“' in pred:
                answer = re.findall(r'“(.*?)”', pred)[0]
            elif '"' in pred:
                answer = re.findall(r'"(.*?)"', pred)[0]
            elif '\'' in pred:
                answer = re.findall(r'\'(.*?)\'', pred)[0]
            else:
                answer = ''
        entry['pred'] = answer
        processed_results.append(entry)
    calculate_accuracy(processed_results)
    with jsonlines.open(save_path, 'w') as w:
        w.write_all(processed_results)


def test_chatgpt(mode, strategy):
    if mode == 'qa':
        template = qa_template_chatgpt
        save_path = f'./results/chatgpt_qa.jsonl'
    elif mode == 'mc':
        template = mc_template_chatgpt
        save_path = f'./results/chatgpt_mc_{strategy}.jsonl'
    else:
        raise ValueError
    if mode == 'qa':
        test_data = [entry for entry in riddle_list if entry['split'] == 'test']
    elif mode == 'mc':
        with jsonlines.open(f'../data/test_{strategy}.jsonl') as r:
            test_data = [entry for entry in r]
    else:
        raise ValueError
    with jsonlines.open(save_path, 'a') as w:
        pass
    with jsonlines.open(save_path) as r:
        tested = [entry['id'] for entry in r]
    test_data = [entry for entry in test_data if entry['id'] not in tested]

    complete_set(test_set=test_data, template=template, save_path=save_path)


def postprocess_chatgpt(mode, strategy):
    if mode == 'qa':
        save_path = f'./results/chatgpt_qa.jsonl'
        strategy = ''
    elif mode == 'mc':
        save_path = f'./results/chatgpt_mc_{strategy}.jsonl'
    else:
        raise ValueError
    with jsonlines.open(save_path) as r:
        results = [entry for entry in r]
    processed_results = []
    for entry in tqdm(results):
        pred = entry['pred']
        answer = ''
        if mode == 'qa':
            if len(pred) == 1:
                answer = pred
            else:
                pred = re.sub('（.*）', '', pred)
                pred = re.sub('\(.*\)', '', pred)
                if '：' in pred:
                    pred = pred.split('：')[1]
                if ':' in pred:
                    pred = pred.split(':')[1]
                answer = pred.strip()
        elif mode == 'mc':
            if len(pred) == 1:
                answer = pred
            else:
                if '：' in pred:
                    pred = pred.split('：')[1]
                if '。' in pred:
                    pred = pred.split('。')[0]
                answer = pred[0]
        entry['pred'] = answer
        processed_results.append(entry)
    print(f'ChatGPT {mode} {strategy} results:')
    calculate_accuracy(processed_results)
    with jsonlines.open(save_path, 'w') as w:
        w.write_all(processed_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chatglm')
    parser.add_argument('--model_path', type=str, default='THUDM/chatglm-6b')
    parser.add_argument('--mode', type=str, default='mc')
    parser.add_argument('--strategy', type=str, default='top')
    args = parser.parse_args()
    model = args.model
    model_path = args.model_path
    mode = args.mode
    strategy = args.strategy
    if mode not in ['qa', 'mc']:
        raise ValueError
    if strategy not in ['top', 'random']:
        raise ValueError
    if model == 'chatglm':
        test_chatglm(mode, strategy, model_path)
        postprocess_chatglm(mode, strategy)
    elif model == 'chatgpt':
        openai.api_key = os.getenv("OPENAI_API_KEY")
        test_chatgpt(mode, strategy)
        postprocess_chatgpt(mode, strategy)


if __name__ == "__main__":
    main()

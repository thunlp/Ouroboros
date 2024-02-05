from datasets import load_dataset, load_metric, load_from_disk

from transformers import AutoTokenizer

from dualdec import dualdec
from dualdec.models import LlamaForCausalLM
from dualdec.cache_engine import CacheEngine

import time, torch, re

import argparse

from tqdm import tqdm

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"

def create_demo_text(n_shot=8, cot_flag=True):
    # This function is borrowed from repo Guangxuan-Xiao/GSM8K-eval
    # https://github.com/Guangxuan-Xiao/GSM8K-eval/
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    index_list = list(range(len(question)))

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text

def clean_answer(model_pred):
    # This function is borrowed from repo Guangxuan-Xiao/GSM8K-eval
    # https://github.com/Guangxuan-Xiao/GSM8K-eval/
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return "[invalid]"

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def build_prompt(input_text, n_shot=8, cot_flag=True):
    # This function is borrowed from repo Guangxuan-Xiao/GSM8K-eval
    # https://github.com/Guangxuan-Xiao/GSM8K-eval/
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def extract_answer_from_output(completion):
    # This function is borrowed from repo Guangxuan-Xiao/GSM8K-eval
    # https://github.com/Guangxuan-Xiao/GSM8K-eval/
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
def is_correct(model_answer, answer):
    # This function is borrowed from repo Guangxuan-Xiao/GSM8K-eval
    # https://github.com/Guangxuan-Xiao/GSM8K-eval/
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def evaluate(model, dataset):
    result = []
    # cnt = 0
    for entry in tqdm(dataset):
        input_str = build_prompt(entry['question'])
        output_str = model.run(input_str)
        pred_ans = clean_answer(output_str)
        is_cor = is_correct(pred_ans, entry['answer'])
        result.append(is_cor)
        # if cnt == 2:
        #     break
        # cnt += 1
    
    return sum(result) / len(result), model.latency()


class AutoRegressiveModel:
    def __init__(self, target_model, tokenizer, max_len, model_type):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.tot_time = 0
        self.tot_tokens = 0
        self.max_len = max_len
        self.model_type = model_type
    
    def run(self, prompt, temperature = 1.0):
        if self.model_type == "deepseek":
            messages=[
                { 'role': 'user', 'content': prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
        prompt_len = input_ids.shape[-1]
        beg_time = time.time()
        output = self.target_model.generate(input_ids, max_new_tokens=self.max_len, do_sample=False)
        end_time = time.time()
        output_len = output.shape[-1]
        output = output[:,prompt_len:]
        self.tot_time += end_time - beg_time
        self.tot_tokens += output_len - prompt_len
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def latency(self):
        return self.tot_tokens / (self.tot_time)


class SyldModel:
    def __init__(self, draft_model, target_model, tokenizer, test_func, max_len, gamma, window_size, guess_set_size, lookahead_level, model_type):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.tot_time = 0
        self.tot_tokens = 0
        self.test_func = test_func
        self.gamma = gamma
        self.window_size = window_size
        self.guess_set_size = guess_set_size
        self.lookahead_level = lookahead_level
        self.max_len = max_len
        self.ngram_cache = CacheEngine(self.lookahead_level, self.guess_set_size)
        self.model_type = model_type
    
    def run(self, prompt, temperature = 1.0):
        # self.ngram_cache = CacheEngine(self.lookahead_level, self.guess_set_size)
        if self.model_type == "deepseek":
            messages=[
                { 'role': 'user', 'content': prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
        prompt_len = input_ids.shape[-1]
        beg_time = time.time()
        output = self.test_func(input_ids, self.draft_model, self.target_model, self.ngram_cache, self.max_len, self.gamma, self.window_size, self.guess_set_size, self.lookahead_level)
        end_time = time.time()
        output_len = output.shape[-1]
        output = output[:,prompt_len:]
        self.tot_time += end_time - beg_time
        self.tot_tokens += output_len - prompt_len
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def latency(self):
        return self.tot_tokens / (self.tot_time)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dualdec', action='store_true', help='Turn on Syld.')
    parser.add_argument('--target_model', type=str, help='Model name or path of target model in both greedy mode or dualdec mode.')
    parser.add_argument('--draft_model', type=str, help='Model name or path of draft model only in dualdec mode.')
    parser.add_argument('--generate_len', type=int, help='Generate length during testing', default=256) 
    parser.add_argument('--data_path', type=str, help="Data path of the dataset")
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--guess_set_size', type=int, default=15)
    parser.add_argument('--lookahead_level', type=int, default=6)
    parser.add_argument('--model_type', type=str, default='llama')
    args = parser.parse_args()  
    return args


def main():
    args = parse()
    if args.dualdec:
        small_model = LlamaForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.float16, device_map='auto')
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto')
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = SyldModel(draft_model=small_model, target_model=target_model, tokenizer=tokenizer, test_func=dualdec, max_len=args.generate_len, gamma=args.gamma, window_size=args.window_size, guess_set_size=args.guess_set_size, lookahead_level=args.lookahead_level, model_type=args.model_type)
    else:
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto', model_type=args.model_type)
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = AutoRegressiveModel(target_model=target_model, tokenizer=tokenizer, max_len=args.generate_len, model_type=args.model_type)


    dataset = load_from_disk(args.data_path)
    print("warm up...")

    for i in range(5):
        model.run("warm up")
    result, latency = evaluate(model, dataset)
    print(result)
    print(f"total speed: {latency:.2f} tok / s")
    return


if __name__ == "__main__":
    main()
'''
Yi:
    greedy: python test_gsm8k.py --target_model <target_model_path> --data_path <data_path>
    dualdec:   python test_gsm8k.py --target_model <target_model_path> --draft_model <draft_model_path> --dualdec --data_path <data_path>
    
Llama-70b:
    greedy: python test_gsm8k.py --target_model <target_model_path> --data_path <data_path>
    dualdec:   python test_gsm8k.py --target_model <target_model_path> --draft_model <draft_model_path> --dualdec --data_path <data_path> --gamma 6 --lookahead_level 6 --window_size 17 --guess_set_size 17
'''
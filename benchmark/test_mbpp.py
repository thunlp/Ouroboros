from tqdm import tqdm

from mbpp.data import write_jsonl, stream_jsonl
from mbpp.evaluation import evaluate_functional_correctness

from transformers import AutoTokenizer

from dualdec import dualdec
from dualdec.models import LlamaForCausalLM
from dualdec.cache_engine import CacheEngine

import time, torch, re

import argparse

from datasets import load_dataset, load_from_disk

def entry_point(
    problems,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problems
    )

    return results


def filter_code(completion: str) -> str:
    pos = completion.find("[DONE]")
    if pos != -1:
        return completion[:pos]
    else:
        return completion


def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))

def gen_context_prompt(entry):
    return f"You are an expert Python programmer, and here is your task: {entry['text']} Your code should pass these tests:\n\n{entry['test_list']}\n[BEGIN]\n{entry['code']}\n[DONE]\n"

def gen_context(dataset, example_num):
    dataset = list(dataset)[:example_num]
    output = ""
    for entry in dataset:
        output += gen_context_prompt(entry)
    return output
        

def format_test_example(q, tests, code: str=None):
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt

def deepseek_gen_context(dataset, example_num):
    examples_str = []
    dataset = list(dataset)
    for i in range(example_num):
        ex = dataset[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]
    return examples_str

def deepseek_convert_for_evaluation(example):
    gpt_completion = example
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example = generation
    return example

def evaluate(model, dataset, args, **kwargs) -> dict:

    example_num = 3

    if args.model_type == 'deepseek' or args.model_type == "codellama":
        context = deepseek_gen_context(dataset['prompt'], example_num)
    else: 
        context = gen_context(dataset['prompt'], example_num)

    dataset = dataset['validation']

    samples = []
    progress_bar = tqdm(total=len(dataset), desc="Generating samples")

    # cnt = 0
    for entry in dataset:
        if args.model_type == "deepseek":
            prompt = '''
                        Please refer the given examples and generate a python function for my problem.
                        Examples are listed as follows:
                        {}

                        Here is my problem:
                        {}
                        '''.strip().format('\n\n'.join(context), format_test_example(entry['text'], entry['test_list'], code=None))
        elif args.model_type == "codellama":
            prompt = '''
                        B_INST\n
                        Please refer the given examples and generate a python function for my problem.
                        Examples are listed as follows:
                        {}

                        Here is my problem:
                        {}\nE_INST
                        '''.strip().format('\n\n'.join(context), format_test_example(entry['text'], entry['test_list'], code=None))
        else:
            prompt = context + f"You are an expert Python programmer, and here is your task: {entry['text']} Your code should pass these tests:\n\n{entry['test_list']}\n[BEGIN]\n"
        comp = model.run(prompt)
        if args.model_type == "deepseek":
            completion = deepseek_convert_for_evaluation(comp)
        elif args.model_type == "codellama":
            completion = re.findall(r'\[PYTHON](.*?)\[/PYTHON]', comp, re.DOTALL)
            if len(completion) > 0:
                completion = completion[0]
            else:
                completion = ""
        else:
            completion = filter_code(comp)
        sample = dict(task_id=entry['task_id'], completion=completion)
        samples.append(sample)
        progress_bar.update(1)
        # if cnt == 1:
        #     break
        # cnt += 1
    progress_bar.close()

    result = None

    pred_filename = "mbpp_predictions.jsonl"
    write_jsonl(pred_filename, samples)

    print("Evaluating...")
    result = entry_point(list(dataset), sample_file=pred_filename)
    return result, model.latency()

class AutoRegressiveModel:
    def __init__(self, target_model, tokenizer, max_len, model_type):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.tot_time = 0
        self.tot_tokens = 0
        self.max_len = max_len
        self.model_type = model_type
    
    def run(self, prompt, temperature = 1.0):
        if self.model_type == 'deepseek' or self.model_type == 'codellama':
            messages=[
                { 'role': 'user', 'content': prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
            attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[-1]
        beg_time = time.time()
        output = self.target_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_len, do_sample=False)
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
    def __init__(self, draft_model, target_model, tokenizer, test_func, max_len, gamma, window_size, guess_set_size, lookahead_level, eos_token_id, model_type):
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
        self.eos_token_id = eos_token_id
        self.ngram_cache = CacheEngine(lookahead_level, guess_set_size)
        self.model_type = model_type
    
    def run(self, prompt, temperature = 1.0):
        # self.ngram_cache = CacheEngine(self.lookahead_level, self.guess_set_size)
        if self.model_type == 'deepseek' or self.model_type == 'codellama':
            messages=[
                { 'role': 'user', 'content': prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
        prompt_len = input_ids.shape[-1]
        beg_time = time.time()
        output = self.test_func(input_ids, self.draft_model, self.target_model, self.ngram_cache, self.max_len, self.gamma, self.window_size, self.guess_set_size, self.lookahead_level, self.eos_token_id)
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
    parser.add_argument('--data_path', type=str, help="Data path of the dataset", default=None)
    parser.add_argument('--generate_len', type=int, help='Generate length during testing', default=512) 
    parser.add_argument('--gamma', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=18)
    parser.add_argument('--guess_set_size', type=int, default=18)
    parser.add_argument('--lookahead_level', type=int, default=5)
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
        model = SyldModel(draft_model=small_model, target_model=target_model, tokenizer=tokenizer, test_func=dualdec, max_len=args.generate_len, gamma=args.gamma, window_size=args.window_size, guess_set_size=args.guess_set_size, lookahead_level=args.lookahead_level, eos_token_id=tokenizer.eos_token_id, model_type=args.model_type)
    else:
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto')
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = AutoRegressiveModel(target_model=target_model, tokenizer=tokenizer, max_len=args.generate_len, model_type=args.model_type)

    if 'deepseek' in args.target_model:
        print("pad_token_id set")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.data_path:
        dataset = load_from_disk(args.data_path)
    else:
        dataset = load_dataset('mbpp')
    
    # for w in range(14, 23):
    #     print(w)
    #     model.window_size = w
    #     model.guess_set_size = w
    #     model.ngram_cache = CacheEngine(args.lookahead_level, w)
    #     model.tot_tokens = 0
    #     model.tot_time = 0

    print("warm up...")
    for i in range(0):
        model.run("warm up")

    # test mbpp
    result, latency = evaluate(model, dataset, args)
    print(result)
    print(f"total speed: {latency:.2f} tok / s")


if __name__ == "__main__":
    main()

'''
Yi:
    greedy: python test_mbpp.py --target_model <target_model_path> --data_path <data_path>
    dualdec:   python test_mbpp.py --target_model <target_model_path> --data_path <data_path> --draft_model <draft_model_path> --dualdec --gamma 12

Deepseek:
    greedy: python test_mbpp.py --target_model <target_model_path> --data_path <data_path> --model_type deepseek
    dualdec:   python test_mbpp.py --target_model <target_model_path> --data_path <data_path> --draft_model <draft_model_path> --dualdec --model_type deepseek --gamma 7 --window_size 15 --guess_set_size 15 --lookahead_level 5

CodeLlama:
    greedy: python test_mbpp.py --target_model <target_model_path> --data_path <data_path> --model_type codellama
    dualdec:   python test_mbpp.py --target_model <target_model_path> --draft_model <draft_model_path> --data_path <data_path> --dualdec --model_type codellama --gamma 8 --lookahead_level 6 --window_size 16 --guess_set_size 16


'''
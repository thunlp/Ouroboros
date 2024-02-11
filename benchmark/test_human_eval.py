from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from transformers import AutoTokenizer

from ouroboros import ouroboros
from ouroboros.models import LlamaForCausalLM
from ouroboros.cache_engine import CacheEngine

import time, torch

import argparse, re


def get_function_name(question: str, lang: str = 'Python'):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix


def extract_generation_code(question, output, verbose: bool=False):
    setting = {
        'full_name': 'Python',
        'indent': 4,
    }
    lang = setting['full_name']
    indent = setting['indent']

    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        
        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1
            
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]
    
        generation = func_prefix + '\n' + body + '\n'
        from IPython import embed; embed()
        result = generation

    except Exception as ex:
        result = question + '\n' + output
    
    return result


def entry_point(
    problem_file: str,
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
        sample_file, k, n_workers, timeout, problem_file
    )

    return results


def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def gen_prompt(prompt: str, args) -> str:
#     if args.model_type == "deepseek":
#         return '''
# Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
# ```{}
# {}
# ```
# '''.strip().format("Python", prompt.strip())
    prompt = (
        "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
        + prompt
    )
    return prompt


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


def evaluate(model, data_path: str, args, **kwargs) -> dict:
    dataset = read_problems(data_path)
    n_sample = kwargs.get("n_sample", 1)
    # best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}
    best_temperature = {1: 0.9, 10: 0.6, 100: 0.8}
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    # cnt = 0
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = gen_prompt(prompt, args)
            temperature = best_temperature[n_sample]
            if temperature > 0:
                completion = model.run(prompt, temperature)
            else:
                completion = model.run(prompt, )
            completion = fix_indents(completion)
            sample = dict(task_id=task_id, completion=filter_code(completion))
            # if i == 0:
            #     print("Prompt: ", "-" * 100)
            #     print(prompt)
            #     print("Completion: ", "-" * 100)
            #     print(filter_code(completion))
            samples.append(sample)
            progress_bar.update(1)
        # if cnt == 2:
        #     break
        # cnt += 1
    progress_bar.close()

    result = None

    pred_filename = f"humaneval_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
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
        if self.model_type == 'deepseek':
            input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
            prompt = input_str
        elif self.model_type == 'codellama':
            prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]
        enc = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_ids = enc['input_ids']
        prompt_len = input_ids.shape[-1]
        beg_time = time.time()
        output = self.target_model.generate(**enc, max_new_tokens=self.max_len, do_sample=False)
        end_time = time.time()
        output_len = output.shape[-1]
        output = output[:,prompt_len:]
        self.tot_time += end_time - beg_time
        self.tot_tokens += output_len - prompt_len
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def latency(self):
        return self.tot_tokens / (self.tot_time)
    

deepseek_temp = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{prompt}\n### Response:\n{prefix}"


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
        if self.model_type == 'deepseek':
            input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
            prompt = input_str
        elif self.model_type == 'codellama':
            prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]
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
    parser.add_argument('--ouroboros', action='store_true', help='Turn on Syld.')
    parser.add_argument('--target_model', type=str, help='Model name or path of target model in both greedy mode or ouroboros mode.')
    parser.add_argument('--draft_model', type=str, help='Model name or path of draft model only in ouroboros mode.')
    parser.add_argument('--data_path', type=str, help="Data path of the dataset")
    parser.add_argument('--generate_len', type=int, help='Generate length during testing', default=512) 
    parser.add_argument('--gamma', type=int, default=12)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--guess_set_size', type=int, default=20)
    parser.add_argument('--lookahead_level', type=int, default=7)
    parser.add_argument('--model_type', type=str, default='llama')
    args = parser.parse_args()  
    return args


def main():
    args = parse()
    if args.ouroboros:
        small_model = LlamaForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.float16, device_map='auto')
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto')
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = SyldModel(draft_model=small_model, target_model=target_model, tokenizer=tokenizer, test_func=ouroboros, max_len=args.generate_len, gamma=args.gamma, window_size=args.window_size, guess_set_size=args.guess_set_size, lookahead_level=args.lookahead_level, eos_token_id=tokenizer.eos_token_id, model_type=args.model_type)
    else:
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto')
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = AutoRegressiveModel(target_model=target_model, tokenizer=tokenizer, max_len=args.generate_len, model_type=args.model_type)

    if 'deepseek' in args.target_model:
        print("pad_token_id set")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("warm up...")
    for i in range(5):
        model.run("warm up")

    # test human_eval
    result, latency = evaluate(model, args.data_path, args)
    print(result)
    print(f"total speed: {latency:.2f} tok / s")


if __name__ == "__main__":
    main()


'''
Yi:
    greedy: python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path>
    ouroboros:   python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path> --draft_model <draft_model_name_or_path> --ouroboros

Deepseek:
    greedy: python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path> --model_type deepseek
    ouroboros:   python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path> --draft_model <draft_model_name_or_path> --ouroboros --gamma 11 --model_type deepseek

CodeLlama:
    greedy: python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path> 
    ouroboros:   python test_human_eval.py --target_model <target_model_name_or_path> --data_path <data_path> --draft_model <draft_model_name_or_path> --ouroboros --model_type codellama --gamma 10 --lookahead_level 6
    

'''
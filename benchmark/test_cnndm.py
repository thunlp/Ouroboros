from datasets import load_dataset, load_metric, load_from_disk

from transformers import AutoTokenizer

from dualdec import dualdec
from dualdec.models import LlamaForCausalLM
from dualdec.cache_engine import CacheEngine

import time, torch

import argparse

from tqdm import tqdm


prompt = "Please summarize the following document within one sentence. Document: {article}  Summary: "

prompt_chat = "[INST] Please summarize the following document within one sentence and do not output anything more. Document: {article}  Summary: [/INST]"


def evaluate(model, dataset, args):
    rouge = load_metric('rouge')
    predicted_summary = []
    std_summary = []
    # cnt = 0
    for entry in tqdm(dataset):
        if args.model_type == "chat":
            input_str = prompt_chat.format(**entry)
        else:
            input_str = prompt.format(**entry)
        torch.cuda.empty_cache()
        output_str = model.run(input_str)
        predicted_summary.append(output_str)
        std_summary.append(entry['highlights'])
        # if cnt == 2:
        #     break
        # cnt += 1
    
    results = rouge.compute(predictions=predicted_summary, references=std_summary)
    pretty_results = [(key, results[key].mid.fmeasure) for key in results.keys()]
    return pretty_results, model.latency()



class AutoRegressiveModel:
    def __init__(self, target_model, tokenizer, max_len):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.tot_time = 0
        self.tot_tokens = 0
        self.max_len = max_len
    
    def run(self, prompt, temperature = 1.0):
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
    def __init__(self, draft_model, target_model, tokenizer, test_func, max_len, gamma, window_size, guess_set_size, lookahead_level):
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
        self.ngram_cache = CacheEngine(lookahead_level, guess_set_size)
    
    def run(self, prompt, temperature = 1.0):
        # self.ngram_cache = CacheEngine(self.lookahead_level, self.guess_set_size)
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
    parser.add_argument('--generate_len', type=int, help='Generate length during testing', default=128) 
    parser.add_argument('--data_path', type=str, help="Data path of the dataset")
    parser.add_argument('--gamma', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--guess_set_size', type=int, default=15)
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
        model = SyldModel(draft_model=small_model, target_model=target_model, tokenizer=tokenizer, test_func=dualdec, max_len=args.generate_len, gamma=args.gamma, window_size=args.window_size, guess_set_size=args.guess_set_size, lookahead_level=args.lookahead_level)
    else:
        target_model = LlamaForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map='auto')
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        model = AutoRegressiveModel(target_model=target_model, tokenizer=tokenizer, max_len=args.generate_len)


    dataset = load_from_disk(args.data_path)
    print("warm up...")

    # for w in range(3, 16):
    #     print(w)
    #     model.gamma = w
    #     # model.guess_set_size = w
    #     # model.window_size = w
    #     # model.ngram_cache = CacheEngine(args.lookahead_level, w)
    #     model.tot_tokens = 0
    #     model.tot_time = 0 
    #     for i in range(5):
    #         model.run("warm up")
    #     result, latency = evaluate(model, dataset, args)
    #     print(result)
    #     print(f"total speed: {latency:.2f} tok / s")
    
    # print("=" * 50)

    # for w in range(3, 9):
    #     print(w)
    #     model.lookahead_level = w
    #     # model.guess_set_size = w
    #     # model.window_size = w
    #     model.ngram_cache = CacheEngine(w, args.guess_set_size)
    #     model.tot_tokens = 0
    #     model.tot_time = 0 
    #     for i in range(5):
    #         model.run("warm up")
    #     result, latency = evaluate(model, dataset, args)
    #     print(result)
    #     print(f"total speed: {latency:.2f} tok / s")
            
    # print("=" * 50)

    # for w in range(14, 23):
    #     print(w)
    #     model.guess_set_size = w
    #     model.window_size = w
    #     model.ngram_cache = CacheEngine(args.lookahead_level, w)
    #     model.tot_tokens = 0
    #     model.tot_time = 0 
    for i in range(5):
        model.run("warm up")
    result, latency = evaluate(model, dataset, args)
    print(result)
    print(f"total speed: {latency:.2f} tok / s")
    return


if __name__ == "__main__":
    main()

'''
Yi-34b:
    greedy: python test_cnndm.py --target_model <target_model_path> --data_path <data_path>
    dualdec: python test_cnndm.py --target_model <target_model_path> --draft_model <draft_model_path> --data_path <data_path> --dualdec

Llama-70b:
    greedy: python test_cnndm.py --target_model <target_model_path> --data_path <data_path> --model_type chat
    dualdec:   python test_cnndm.py --target_model <target_model_path> --draft_model <draft_model_path> --data_path <data_path> --dualdec --window_size 13 --guess_set_size 13 --lookahead_level 5 --gamma 4 --model_type chat
'''
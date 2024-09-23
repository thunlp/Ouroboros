"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file

from model.ouroboros import ouroboros_specbench
from model.ouroboros.models import LlamaForCausalLM
from model.ouroboros.cache_engine import CacheEngine
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--target-model", type=str, default="/data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--target-model", type=str, required=True)

    # parser.add_argument("--draft-model", type=str, default="/data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--draft-model", type=str, required=True)


    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = LlamaForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )
    draft_model = LlamaForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )
    # draft_model = torch.compile(draft_model)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    assert args.temperature == 0, "Currently only support greedy decoding"

    ngram_cache = CacheEngine(5, 15)
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ouroboros_specbench,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_steps=args.max_steps,
        draft_model=draft_model,
        ngram_cache=ngram_cache,
    )

    reorg_answer_file(answer_file)

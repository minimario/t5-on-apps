import os
import argparse
import torch

from typing import List, Union

# Load model
from transformers import (
    T5Config, 
    T5ForConditionalGeneration, 
    AutoTokenizer,
    AutoModelForCausalLM,
)

from reindent import reindent_code

cache_dir = "/scratch/gua"
# cache_dir = "~/.cache/huggingface/transformers/"

def load_model(model_name="codet5", device="cuda"):
    if model_name == "codet5":
        config = T5Config.from_json_file("codet5-config/config.json")
        model = T5ForConditionalGeneration.from_pretrained("codet5-config/pytorch_model.bin", config=config, cache_dir=cache_dir)
    elif model_name == "gpt-neo-125M":
        model = AutoModelForCausalLM.from_pretrained("Guangxuan-Xiao/gpt-neo-125M-apps", cache_dir=cache_dir)
    elif model_name == "gpt-neo-1.3B":
        model = AutoModelForCausalLM.from_pretrained("Guangxuan-Xiao/gpt-neo-1.3B-apps", cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown model {model_name}")
    if device == "cuda":
        model.cuda()
    model.eval()
    return model

def load_tokenizer(model_name="codet5"):
    if model_name == "codet5":
        return AutoTokenizer.from_pretrained("Salesforce/codet5-large", cache_dir=cache_dir)
    elif model_name == "gpt-neo-125M":
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", cache_dir=cache_dir)
    elif model_name == "gpt-neo-1.3B":
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown model {model_name}")

def load_model_and_tokenizer(model_name="codet5", device="cuda"):
    model = load_model(model_name, device)
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer

# Load train and val datasets
from datasets import load_dataset

def load_orig_dataset():
    return load_dataset("codeparrot/apps", split="train", cache_dir=cache_dir)

def load_datasets():
    orig_train_dataset = load_dataset("codeparrot/apps", split="train", cache_dir=cache_dir)

    # prepare val dataset: only problems that Codex gets right
    val_indices = [26, 30, 35, 51, 54, 76, 103, 127, 139, 158, 159, 161, 170, 178, 182, 184, 187, 188, 193, 200, 201, 203, 205, 207, 211, 215, 226, 228, 234, 236, 240, 252, 260, 261, 264, 272, 275, 276, 279, 295, 303, 304, 305, 309, 310, 313, 329, 333, 343, 351, 354, 357, 370, 372, 379, 383, 384, 386, 394, 407, 410, 418, 443, 447, 450, 453, 454, 467, 468, 469, 473, 474, 485, 488, 489, 526, 527, 528, 537, 538, 539, 540, 543, 557, 560, 566, 572, 587, 594, 603, 612, 614, 615, 623, 635, 647, 653, 661, 667, 687, 689, 691, 702, 703, 704, 707, 712, 721, 746, 750, 763, 767, 796, 814, 823, 831, 832, 833, 839, 847, 849, 859, 878, 879, 880, 895, 916, 930, 931, 940, 942, 947, 948, 953, 954, 963, 969, 979, 993, 995, 1002, 1016, 1018, 1037, 1038, 1039, 1045, 1050, 1053, 1087, 1092, 1100, 1111, 1120, 1126, 1129, 1141, 1148, 1155, 1167, 1172, 1174, 1178, 1193, 1195, 1200, 1211, 1222, 1230, 1242, 1243, 1248, 1254, 1265, 1298, 1306, 1307, 1318, 1319, 1321, 1335, 1346, 1360, 1371, 1375, 1380, 1382, 1398, 1402, 1412, 1417, 1421, 1423, 1427, 1452, 1457, 1459, 1460, 1472, 1485, 1490, 1497, 1499, 1514, 1516, 1518, 1540, 1545, 1546, 1550, 1554, 1562, 1564, 1567, 1581, 1590, 1601, 1609, 1624, 1628, 1631, 1641, 1659, 1662, 1669, 1674, 2045, 2048, 2071, 2115, 2117, 2118, 2120, 2126, 2128, 2133, 2149, 2169, 2174, 2177, 2191, 2193, 2197, 2200, 2218, 2221, 2230, 2245, 2249, 2255, 2268, 2272, 2273, 2284, 2296, 2303, 2307, 2311, 2312, 2324, 2327, 2331, 2333, 2336, 2363, 2366, 2370, 2403, 2405, 2413, 2423, 2431, 2432, 2442, 2446, 2450, 2458, 2464, 2468, 2473, 2478, 2501, 2502, 2513, 2518, 2527, 2532, 2533, 2552, 2553, 2555, 2556, 2558, 2562, 2566, 2569, 2571, 2577, 2582, 2583, 2588, 2595, 2596, 2598, 2606, 2610, 2619, 2623, 2626, 2632, 2633, 2644, 2646, 2648, 2650, 2653, 2655, 2656, 2660, 2664, 2676, 2682, 2683, 2694, 2700, 2709, 2710, 2714, 2716, 2728, 2733, 2742, 2755, 2776, 2782, 2784, 2791, 2800, 2820, 2826, 2834, 2846, 2852, 2853, 2866, 2883, 2884, 2924, 2935, 2945, 2952, 2954, 2956, 2957, 2959, 2961, 2965, 2967, 2977, 2978, 2979, 2981, 2989, 2996, 2998, 3010, 3013, 3015, 3030, 3032, 3043, 3044, 3055, 3057, 3061, 3070, 3110, 3113, 3117, 3118, 3132, 3136, 3142, 3144, 3146, 3155, 3157, 3159, 3177, 3179, 3183, 3184, 3190, 3196, 3198, 3213, 3214, 3229, 3237, 3248, 3249, 3296, 3301, 3305, 3333, 3338, 3344, 3346, 3350, 3365, 3374, 3383, 3402, 3413, 3422, 3427, 3441, 3449, 3452, 3455, 3461, 3463, 3464, 3469, 3471, 3473, 3480, 3482, 3496, 3501, 3504, 3509, 3521, 3543, 3550, 3553, 3564, 3566, 3570, 3593, 3603, 3604, 3629, 3632, 3638, 3639, 3643, 3649, 3665, 3668, 3672, 3676, 3692, 3707, 3715, 3717, 3724, 3727, 3731, 3743, 3746, 3753, 3766, 3782, 3790, 3793, 3807, 3818, 3821, 3824, 3825, 3831, 3833, 3836, 3839, 3850, 3864, 3865, 3867, 3876, 3889, 3890, 3892, 3895, 3896, 3907, 3908, 3916, 3923, 3927, 3931, 3937, 3952, 3962, 3965, 3968, 3972, 3974, 3976, 3979, 3988, 3994, 3997, 3999, 4010, 4014, 4019, 4021, 4023, 4029, 4034, 4036, 4038, 4063, 4073, 4074, 4080, 4085, 4092, 4095, 4096, 4100, 4110, 4138, 4139, 4153, 4158, 4174, 4180, 4192, 4193, 4225, 4238, 4255, 4258, 4284, 4292, 4297, 4303, 4315, 4321, 4323, 4329, 4335, 4336, 4338, 4341, 4352, 4370, 4371, 4373, 4381, 4386, 4397, 4398, 4402, 4415, 4417, 4424, 4425, 4430, 4439, 4447, 4449, 4455, 4464, 4491, 4504, 4519, 4524, 4535, 4536, 4537, 4538, 4544, 4559, 4560, 4567, 4575, 4585, 4591, 4594, 4600, 4603, 4613, 4615, 4616, 4618, 4624, 4640, 4644, 4648, 4663, 4671, 4679, 4685, 4696, 4712]
    val_dataset = orig_train_dataset.select(val_indices)

    # prepare train dataset
    train_indices_all = [i for i in range(len(orig_train_dataset)) if i not in val_indices]
    train_dataset_all = orig_train_dataset.select(train_indices_all)

    # prepare train dataset but only with problems that have tests
    train_indices_with_tests = []
    for i in range(len(train_dataset_all)):
        try:
            num_tests = len(json.loads(train_dataset_all[i]["input_output"])["inputs"])
            if num_tests > 0:
                train_indices_with_tests.append(i)
        except:
            pass
    train_dataset_with_tests = train_dataset_all.select(train_indices_with_tests)

    assert len(val_dataset) == 598
    assert len(train_dataset_all) == 4402
    assert len(train_dataset_with_tests) == 3851

    # prepare test dataset
    test_dataset = load_dataset("codeparrot/apps", split="test", cache_dir=cache_dir)
    return train_dataset_all, train_dataset_with_tests, val_dataset, test_dataset

import json
def get_problem_parts(sample, mode = "train", answer_type_preprocessing = "loubna_corrected"):
    question_str = sample["question"]
    starter_code = "" if len(sample["starter_code"]) == 0 else sample["starter_code"]
    # if answer_type_preprocessing != "loubna":
    #     starter_code = reindent_code(starter_code)

    if answer_type_preprocessing == "loubna":
        try:
            input_output = json.loads(sample["input_output"])
            if not input_output.get("fn_name"):
                fn_name = None
            else:
                fn_name = input_output["fn_name"]
        except ValueError:
            fn_name = None

        if fn_name:
            answer_type = "\nUse Standard Input format\n"
        else:
            answer_type = "\nUse Call-Based format\n"

    elif answer_type_preprocessing in ["jeevana", "loubna_corrected"]:
        try:
            input_output = json.loads(sample["input_output"])
            if not input_output.get("fn_name"):
                fn_name = None
            else:
                fn_name = input_output["fn_name"]
        except ValueError:
            fn_name = None

        if fn_name:
            answer_type = "\nUse Call-Based format\n"
        else:
            answer_type = "\nUse Standard Input format\n"
    
    elif mode == "test":
        if json.loads(sample["input_output"]).get("fn_name"):
            answer_type = "\nUse Call-Based format\n"
        else:
            answer_type = "\nUse Standard Input format\n"
    else:
        if starter_code == "":
            answer_type = "\nUse Standard Input format\n"
        else:
            answer_type = "\nUse Call-Based format\n" 
    return (question_str, starter_code, answer_type)

def format_problem_str(sample, mode = "train", answer_type_preprocessing = "loubna_corrected"):
    question_str, starter_code, answer_type = get_problem_parts(sample, mode, answer_type_preprocessing)
    if answer_type_preprocessing == "jeevana":
        q_str = f"# QUESTION:\n{question_str}{starter_code}{answer_type}PYTHON CODE:"
        q_str = q_str.replace("\n", "\n# ") + '\n'
    else:
        q_str = f"\nQUESTION:\n{question_str}\n{starter_code}\n{answer_type}\nANSWER:\n"
    return q_str

def format_problem_tokenized(sample, tokenizer, max_length, mode = "train", answer_type_preprocessing = "loubna_corrected", device="cuda"):
    if answer_type_preprocessing == "jeevana":
        raise Exception("Jeevana preprocessing not supported yet")
    question_str, starter_code, answer_type = get_problem_parts(sample, mode, answer_type_preprocessing)
    parts = ["\nQUESTION:\n", question_str, f"\n{starter_code}\n{answer_type}\nANSWER:\n"]
    tokenized_parts = []
    for i in range(len(parts)):
        if i == 0:
            tokenized_parts.append(tokenizer(parts[i])["input_ids"][:-1])
        elif i == len(parts) - 1:
            tokenized_parts.append(tokenizer(parts[i])["input_ids"][1:])
        else:
            tokenized_parts.append(tokenizer(parts[i])["input_ids"][1:-1])
    length_to_truncate = max(0, sum([len(part) for part in tokenized_parts]) - max_length)
    if length_to_truncate > 0:
        tokenized_parts[1] = tokenized_parts[1][:-length_to_truncate]
    input_tokens = tokenized_parts[0] + tokenized_parts[1] + tokenized_parts[2]
    if device == "cuda":
        return torch.LongTensor([input_tokens]).cuda()
    else:
        return torch.LongTensor([input_tokens])


# def generate_candidates_using_solution(model, tokenizer, sample):
#     input_str = format_problem(sample)
#     question_ids = tokenizer(input_str)["input_ids"]

#     solutions = json.loads(sample["solutions"])
#     solution_ids = tokenizer(solutions[0])["input_ids"]

#     # pad_token = [tokenizer.pad_token_id]
#     # question_ids.extend(pad_token)

#     print("Length of solution: ", len(solution_ids))
#     partial_solution_length = int(len(solution_ids) * 0.9)
#     # decoder_start_token_id = solution_ids[20]
#     solution_ids = solution_ids[0:partial_solution_length]
#     # solution_ids = []
#     question_ids.extend(solution_ids)
#     question_ids = question_ids[0:2000]

#     print("Question with partial solution:")
#     # print(question_ids)
#     # print(tokenizer.decode(question_ids))

#     input_ids = torch.LongTensor([question_ids]).cuda()
#     # print("last 10 of input_ids", input_ids[0][-10:])
#     generated_ids = model.generate(input_ids, max_length=1000, num_return_sequences=10, temperature=0.9, do_sample=True)
#     # generated_ids = model.generate(input_ids, decoder_start_token_id=decoder_start_token_id, \
#     #     begin_suppress_tokens=[1], max_length=1000, do_sample=False)
#     # print(generated_ids[0][0:10])
#     candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

#     for i in range(len(candidates)):    
#         if "ANSWER:" in candidates[i]:
#             candidates[i] = candidates[i].split("ANSWER:")[1]
    
#     return candidates

def get_per_candidate_results(results : List[List[Union[int, bool]]]) -> List[str]:
    per_candidate_results = []
    for r in results:
        if r == [True] * len(r) and len(r) != 0:
            per_candidate_results.append("Correct")
        elif -2 in r or -1 in r:
            per_candidate_results.append("Execution error")
        else:
            per_candidate_results.append("Intent error")
    return per_candidate_results

def ternary_to_binary(results : List[str]) -> List[str]:
    conversion = {"Correct": "Correct", "Execution error": "Wrong", "Intent error": "Wrong"}
    return [conversion[r] for r in results]

def get_percentage_correct(results):
    score = 0
    max_score = 0
    for r in results:
        max_score += 1
        if r == [True] * len(r) and len(r) != 0:
            score += 1
    return score / max_score

def generate_candidates(model, tokenizer, sample):
    input_tokens = format_problem_tokenized(sample, tokenizer, max_length=1024)
    all_candidates = []

    tries = 0
    while len(all_candidates) < 20 and tries < 8:
        tries += 1
        try:
            print("successful generation!")
            generated_ids = model.generate(input_tokens, 
                                        max_new_tokens=2000-input_tokens.shape[1], 
                                        num_return_sequences=20, 
                                        temperature=0.9, 
                                        do_sample=True)
            candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for c in candidates:
                print(c)
                print("--")
            all_candidates += candidates
            print(len(all_candidates))
            del generated_ids
            torch.cuda.empty_cache()
        except Exception as e:
            print("failed to generate candidates")
            print(e)
            pass

    for i in range(len(all_candidates)):    
        if "ANSWER:" in all_candidates[i]:
            all_candidates[i] = all_candidates[i].split("ANSWER:")[1]
    return all_candidates


def generate_candidates_using_solution(model, tokenizer, sample, num_solutions=100000, len_percentage=0.9):
    input_str = format_problem(sample)
    question_ids = tokenizer(input_str)["input_ids"]
    solutions = json.loads(sample["solutions"])

    for i in range(min(len(solutions), num_solutions)):
        # print(f"Question\n{input_str}\n{'*'*70+'\n\n'}Solution\n{solutions[i].__repr__()}\n{'*'*70+'\n\n'}")
        solution_ids = tokenizer(solutions[i])["input_ids"]
        partial_solution_length = int(len(solution_ids) * len_percentage)
        solution_ids = solution_ids[0:partial_solution_length]
        question_ids.extend(solution_ids)
        question_ids = question_ids[0:2000]
        # print(f"Partial program:\n{tokenizer.decode(question_ids)}\n{'*'*70+'\n\n'}")
        input_ids = torch.LongTensor([question_ids]).cuda()
        candidates, results, score = get_partial_program_score(model, tokenizer, sample, input_ids)
        # print(f"partial program {i}, score {score}")
    
    return candidates, results, score


def get_partial_program_score(model, tokenizer, sample, input_ids):
    # Note: input_ids is a torch Tensor that has been moved to the GPU
    candidates = []

    for i in range(2):
        generated_ids = model.generate(input_ids, max_length=1000, \
            num_return_sequences=4, temperature=0.9, do_sample=True)
        candidates += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for c in tokenizer.batch_decode(input_ids, skip_special_tokens=False):
            print("input|"+c+"|input")
    generated_ids = model.generate(input_ids, max_length=1000, \
        num_return_sequences=2, temperature=0.9, do_sample=True)
    candidates += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for c in tokenizer.batch_decode(input_ids, skip_special_tokens=False):
        print("input|"+c+"|input")

    for i in range(len(candidates)):    
        if "ANSWER:" in candidates[i]:
            candidates[i] = candidates[i].split("ANSWER:")[1]
    results = evaluate_generations_problem(candidates, sample, debug=False, verbose=False)
    score = get_percentage_correct(results)
    return candidates, results, score

from utils import evaluate_generations_problem
def run_test_cases(model, tokenizer, sample, use_solution = False):
    if use_solution:
        candidates = generate_candidates_using_solution(model, tokenizer, sample)
    else:
        candidates = generate_candidates(model, tokenizer, sample)

    print("Candidates generated!")
    for cand in candidates:
        print(cand)
        print("-"*20)
    results = evaluate_generations_problem(candidates, sample)
    print("Generations evaluated!")
    return results
    # return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--difficulties", type=str, required=True)

    args = parser.parse_args()
    assert args.split in ["train", "val", "test"]

    gpu_id = args.gpu_id
    num_gpu = args.num_gpu
    model_name = args.model_name
    assert model_name in ["gpt-neo-125M", "gpt-neo-1.3B", "codet5"]

    train_dataset_all, train_dataset_with_tests, val_dataset, test_dataset = load_datasets()
    if args.split == "train":
        dataset = train_dataset_with_tests
    elif args.split == "val":
        dataset = val_dataset
    elif args.split == "test":
        dataset = test_dataset
    else: 
        raise Exception("Invalid split")

    print("Datasets loaded!")

    model, tokenizer = load_model_and_tokenizer(model_name)
    print("Model and tokenizer loaded!")

    save_path = {"gpt-neo-125M": "full_programs_125m_new", \
                 "gpt-neo-1.3B": "full_programs_1b_new", \
                 "codet5": f"full_programs_t5_{args.split}"}
    save_dir = save_path[model_name]
    print("Saving to: ", save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # pick the correct samples
    assert args.difficulties in ["introductory", "interview", "competition", "all"]
    if args.difficulties == "all":
        difficulties = ["introductory", "interview", "competition"]
    else:
        difficulties = [args.difficulties]
    samples = [i for i in range(len(dataset)) if dataset[i]["difficulty"] in difficulties]

    # split the samples across the gpus
    num_samples = len(samples)
    samples_per_gpu = (num_samples + num_gpu - 1) // num_gpu
    start = gpu_id * samples_per_gpu
    end = min((gpu_id+1) * samples_per_gpu, num_samples)
    print("processing gpu's", gpu_id, "samples", start, "to", end)

    # begin the generation
    for i in range(start, end):
        sample_i = samples[i]
        save_path = f"{save_dir}/{args.split}_{sample_i}.json"
        if os.path.exists(save_path):
            continue
        try:
            sample = dataset[sample_i]
            candidates = generate_candidates(model, tokenizer, sample)
            results = evaluate_generations_problem(candidates, sample)
            with open(save_path, "w") as f:
                json.dump(candidates, f)
            print(results)
        except Exception as e:
            print(f"failed {sample_i} :(")
            print(e)

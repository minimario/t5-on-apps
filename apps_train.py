"""
Fine-Tune LM on APPS train split
"""

import argparse
import os
import torch

from apps_dataset import APPSBaseDataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="codeparrot/codeparrot-small")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=250, type=int)
    parser.add_argument("--save_freq", default=250, type=int)
    return parser.parse_args()


def get_dataset(dataset, args, mode):

    train_data = APPSBaseDataset(
        dataset=dataset, max_tokens=args.max_length, tokenizer_path=args.model_ckpt, mode=mode,
        answer_type_preprocessing="loubna"
    )
    print("Using loubna preprocessing for answer type")
    return train_data


def run_training(args, train_data, val_data):
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)

    train_data.start_iteration = 0

    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy = "steps",
        num_train_epochs=args.num_epochs,
        max_steps = args.max_steps,
        eval_steps = args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps = args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,

        run_name="apps-train",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    print("Training...")
    trainer.train()

    print("saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):

    dataset = load_dataset("codeparrot/apps", split="train")
    val_indices = [26, 30, 35, 51, 54, 76, 103, 127, 139, 158, 159, 161, 170, 178, 182, 184, 187, 188, 193, 200, 201, 203, 205, 207, 211, 215, 226, 228, 234, 236, 240, 252, 260, 261, 264, 272, 275, 276, 279, 295, 303, 304, 305, 309, 310, 313, 329, 333, 343, 351, 354, 357, 370, 372, 379, 383, 384, 386, 394, 407, 410, 418, 443, 447, 450, 453, 454, 467, 468, 469, 473, 474, 485, 488, 489, 526, 527, 528, 537, 538, 539, 540, 543, 557, 560, 566, 572, 587, 594, 603, 612, 614, 615, 623, 635, 647, 653, 661, 667, 687, 689, 691, 702, 703, 704, 707, 712, 721, 746, 750, 763, 767, 796, 814, 823, 831, 832, 833, 839, 847, 849, 859, 878, 879, 880, 895, 916, 930, 931, 940, 942, 947, 948, 953, 954, 963, 969, 979, 993, 995, 1002, 1016, 1018, 1037, 1038, 1039, 1045, 1050, 1053, 1087, 1092, 1100, 1111, 1120, 1126, 1129, 1141, 1148, 1155, 1167, 1172, 1174, 1178, 1193, 1195, 1200, 1211, 1222, 1230, 1242, 1243, 1248, 1254, 1265, 1298, 1306, 1307, 1318, 1319, 1321, 1335, 1346, 1360, 1371, 1375, 1380, 1382, 1398, 1402, 1412, 1417, 1421, 1423, 1427, 1452, 1457, 1459, 1460, 1472, 1485, 1490, 1497, 1499, 1514, 1516, 1518, 1540, 1545, 1546, 1550, 1554, 1562, 1564, 1567, 1581, 1590, 1601, 1609, 1624, 1628, 1631, 1641, 1659, 1662, 1669, 1674, 2045, 2048, 2071, 2115, 2117, 2118, 2120, 2126, 2128, 2133, 2149, 2169, 2174, 2177, 2191, 2193, 2197, 2200, 2218, 2221, 2230, 2245, 2249, 2255, 2268, 2272, 2273, 2284, 2296, 2303, 2307, 2311, 2312, 2324, 2327, 2331, 2333, 2336, 2363, 2366, 2370, 2403, 2405, 2413, 2423, 2431, 2432, 2442, 2446, 2450, 2458, 2464, 2468, 2473, 2478, 2501, 2502, 2513, 2518, 2527, 2532, 2533, 2552, 2553, 2555, 2556, 2558, 2562, 2566, 2569, 2571, 2577, 2582, 2583, 2588, 2595, 2596, 2598, 2606, 2610, 2619, 2623, 2626, 2632, 2633, 2644, 2646, 2648, 2650, 2653, 2655, 2656, 2660, 2664, 2676, 2682, 2683, 2694, 2700, 2709, 2710, 2714, 2716, 2728, 2733, 2742, 2755, 2776, 2782, 2784, 2791, 2800, 2820, 2826, 2834, 2846, 2852, 2853, 2866, 2883, 2884, 2924, 2935, 2945, 2952, 2954, 2956, 2957, 2959, 2961, 2965, 2967, 2977, 2978, 2979, 2981, 2989, 2996, 2998, 3010, 3013, 3015, 3030, 3032, 3043, 3044, 3055, 3057, 3061, 3070, 3110, 3113, 3117, 3118, 3132, 3136, 3142, 3144, 3146, 3155, 3157, 3159, 3177, 3179, 3183, 3184, 3190, 3196, 3198, 3213, 3214, 3229, 3237, 3248, 3249, 3296, 3301, 3305, 3333, 3338, 3344, 3346, 3350, 3365, 3374, 3383, 3402, 3413, 3422, 3427, 3441, 3449, 3452, 3455, 3461, 3463, 3464, 3469, 3471, 3473, 3480, 3482, 3496, 3501, 3504, 3509, 3521, 3543, 3550, 3553, 3564, 3566, 3570, 3593, 3603, 3604, 3629, 3632, 3638, 3639, 3643, 3649, 3665, 3668, 3672, 3676, 3692, 3707, 3715, 3717, 3724, 3727, 3731, 3743, 3746, 3753, 3766, 3782, 3790, 3793, 3807, 3818, 3821, 3824, 3825, 3831, 3833, 3836, 3839, 3850, 3864, 3865, 3867, 3876, 3889, 3890, 3892, 3895, 3896, 3907, 3908, 3916, 3923, 3927, 3931, 3937, 3952, 3962, 3965, 3968, 3972, 3974, 3976, 3979, 3988, 3994, 3997, 3999, 4010, 4014, 4019, 4021, 4023, 4029, 4034, 4036, 4038, 4063, 4073, 4074, 4080, 4085, 4092, 4095, 4096, 4100, 4110, 4138, 4139, 4153, 4158, 4174, 4180, 4192, 4193, 4225, 4238, 4255, 4258, 4284, 4292, 4297, 4303, 4315, 4321, 4323, 4329, 4335, 4336, 4338, 4341, 4352, 4370, 4371, 4373, 4381, 4386, 4397, 4398, 4402, 4415, 4417, 4424, 4425, 4430, 4439, 4447, 4449, 4455, 4464, 4491, 4504, 4519, 4524, 4535, 4536, 4537, 4538, 4544, 4559, 4560, 4567, 4575, 4585, 4591, 4594, 4600, 4603, 4613, 4615, 4616, 4618, 4624, 4640, 4644, 4648, 4663, 4671, 4679, 4685, 4696, 4712]
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    train_dataset.shuffle(seed=args.seed)
    val_dataset.shuffle(seed=args.seed)

    train_data = get_dataset(train_dataset, args, mode="train")
    val_data = get_dataset(val_dataset, args, mode="val")

    print(f"size of training data {len(train_data)}\nsize of validation data {len(val_data)}")
    run_training(args, train_data, val_data)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
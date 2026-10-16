import argparse

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
parser.add_argument("--resume", type=str, default=None,
                help="Resume from checkpoint. Accepts: (1) .pth file for omni-only (legacy), or (2) directory with omni_parameters.pth and router_parameters.pth")
parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
parser.add_argument("--calib_dataset",type=str,default="wikitext2",
    choices=["wikitext2", "ptb", "c4","c4_new", "mix","pile"],
    help="Where to extract calibration data from.",
)
parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
parser.add_argument("--tasks", default="")
parser.add_argument("--eval_ppl", action="store_true")
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--wbits", type=int, default=2)
parser.add_argument("--attnbits", type=int, default=4)
parser.add_argument("--abits", type=int, default=16)
# parser.add_argument("--slice_bits_list", type=int, nargs="+", default=[2,2,2,2], help="slicing bits from wbits")
parser.add_argument("--slice_bits_list", type=int, nargs="+", help="slicing bits from wbits")
parser.add_argument("--stage_weight", type=float, nargs="+", default=[1.0, 1.0], help="weight for each stage loss")

parser.add_argument("--slice_bits_eval", type=int, default=0)
parser.add_argument("--bitmoe_lr", type=float, default=1e-5)

parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--let_lr", type=float, default=5e-3)
parser.add_argument("--lwc_lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
parser.add_argument("--lwc",default=True, action="store_true",help="activate learnable weight clipping")
parser.add_argument("--aug_loss", default=True, action="store_true", help="calculate additional loss with same input")
parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
parser.add_argument(
    "--attn_implementation",
    type=str, required=False, default="eager",
    choices=["eager", "sdpa", "flash_attention_2"],
    help="attention implementation that the model works with",
)
parser.add_argument("--net", type=str, default=None, choices=net_choices)
parser.add_argument("--act-scales", type=str, default=None)
parser.add_argument("--act-shifts", type=str, default=None)
parser.add_argument("--slicing", default=False, action="store_true", help="use slicing obtian MSBs results")

parser.add_argument("--target_activation_ratio", type=float, default=None, help="target activation ratio for MoE routers: target activation ratio has precedence")
parser.add_argument("--target_bit", type=float, default=None, help="target effective bits for MoE routers: target activation ratio has precedence")
parser.add_argument("--target_activation_ratio_eval", type=float, default=None, help="target activation ratio for MoE routers during evaluation")
parser.add_argument("--target_bit_eval", type=float, default=None, help="target effective bits for MoE routers during evaluation")
parser.add_argument("--reg_weight", type=float, default=1e-5, help="weight for moe regularization")

parser.add_argument("--onlymlp", default=False, action="store_true", help="only quantize MLP (not ATTN) in Llama layers")

# wandb arguments
parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
parser.add_argument("--wandb_project", type=str, default="test", help="W&B project name")
parser.add_argument("--wandb_name", type=str, default="test_run", help="W&B run name")
parser.add_argument("--stage_weight_1", type=float, default=None, help="Stage weight[1] for sweep")
args, unknown = parser.parse_known_args()
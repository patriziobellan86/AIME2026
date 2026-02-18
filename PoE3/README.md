# PoE_small_v2
second version of PoE 



{"name": "test",
"task": "Classify a sentence as in-topic or out-of-topic",
"context": "A supportive chatbot designed for pregnant women and new mothers, offering guidance from pregnancy through the baby’s first 1,000 days. It provides answers on health, baby care, nutrition, psychological support, and financial planning, as well as assistance with related questions like choosing the best hospital for delivery.",
"description_framework": "Persona",
"model": "meta-llama/Llama-3.2-1B",
"output_dir": "/PoEv2/outdir",
"input": "/PoEv2/x.txt",
"temperature": 1.2,
"nucleus": 0.9,
"alternatives":1,
"resume":1,
"cache_dir": "/PoEv2/cache",
"max_experts_number": 3, if this number is less than 1, use unconstrained number of experts
"baseline": 0,
"token": "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"
}

if 'no-personality'  is defined in the args_dict and it is set to true, then the expert agents are created without a personality description.
just the fiedl of expertize is provided.

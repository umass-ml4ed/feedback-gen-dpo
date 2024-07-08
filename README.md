# Goal-Oriented Feedback Generation and Evaluation

This is the code for the paper [Improving the Validity of Automatically Generated Feedback via Reinforcement Learning](https://arxiv.org/abs/2403.01304). We propose a rubric for evaluating math feedback, use GPT-4 to evaluate a dataset of human and LM-generated feedback, and then use direct preference optimization (DPO) to generate higher-scoring feedback messages.

## Running

### Setup
```
python3 -m venv fb_env
source fb_env/bin/activate
python3 -m pip install -r requirements.txt
```

### Construct Dataset
Note that our dataset is private, but we're including these steps for completeness.
```
python3 create_reward_dataset.py --expand
python3 create_reward_dataset.py --generate random --model code-davinci-002
python3 create_reward_dataset.py --generate knn --model code-davinci-002
python3 create_reward_dataset.py --generate zs --model gpt-3.5-turbo
python3 create_reward_dataset.py --compile single
python3 create_reward_dataset.py --subset
python3 create_reward_dataset.py --annotate --model gpt-4
```

### Get Label Statistics and Agreement
```
python3 analyze_reward_dataset.py gpt4 ours --do_bertscore
```

### Train SFT Model
```
python3 train_llm.py --sft --include_rubric --model_name feedback-gen-sft --epochs 3
```

### Train DPO Models
```
python3 train_llm.py --dpo --include_rubric --model_name feedback-gen-dpo-s --pt_model_name feedback-gen-sft --dpo_mmo 0 --dpo_mmi 0 --batch_size 8 --grad_accum_steps 8 --epochs 3
python3 train_llm.py --dpo --include_rubric --model_name feedback-gen-dpo-sm --pt_model_name feedback-gen-sft --dpo_mmo 1 --dpo_mmi 1 --batch_size 8 --grad_accum_steps 8 --epochs 3
```

### Generate Feedback
```
python3 train_llm.py --generate --include_rubric                                             # Zero-Shot
python3 train_llm.py --generate --include_rubric --model_name feedback-gen-sft               # SFT
python3 train_llm.py --generate --include_rubric --model_name feedback-gen-dpo-s             # DPO (Score)
python3 train_llm.py --generate --include_rubric --model_name feedback-gen-dpo-sm            # DPO (Score + Mismatch)
python3 create_reward_dataset.py --generate zs --model gpt-4 --include_rubric --include_sol  # GPT-4
```

### Evaluate Feedback
```
python3 eval_results.py feedback_gen_results_meta-llama-Llama-2-7b-chat-hf_rubric_greedy.csv --metric llm --src ft  # Zero-Shot
python3 eval_results.py feedback_gen_results_feedback-gen-sft_greedy.csv --metric llm --src ft                      # SFT
python3 eval_results.py feedback_gen_results_feedback-gen-dpo-s_greedy.csv --metric llm --src ft                    # DPO (Score)
python3 eval_results.py feedback_gen_results_feedback-gen-dpo-sm_greedy.csv --metric llm --src ft                   # DPO (Score + Mismatch)
python3 eval_results.py data/icl/feedback_test_zs_gpt-4_sol_rubric.csv --metric llm --src icl                       # GPT-4
python3 eval_results.py data/raw/eedi_expanded_test.csv --metric llm --src og                                       # Human
```

## Citation
If you used our code or found this work useful in any way, please cite us!
```
@InProceedings{scarlatos2024improving,
      author="Scarlatos, Alexander and Smith, Digory and Woodhead, Simon and Lan, Andrew",
      editor="Olney, Andrew M. and Chounta, Irene-Angelica and Liu, Zitao and Santos, Olga C. and Bittencourt, Ig Ibert",
      title="Improving the Validity of Automatically Generated Feedback via Reinforcement Learning",
      booktitle="Artificial Intelligence in Education",
      year="2024",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="280--294",
      isbn="978-3-031-64302-6"
}
```

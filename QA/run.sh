mkdir "results"

KEY='your_openai_api_key'

# Embed riddles and calculate similarity
OPENAI_API_KEY=$KEY python embedding.py
# or use the pre-calculated similarity matrix in 'data' folder

# Retrieval QA
CUDA_VISIBLE_DEVICES=0 python match.py \
  --model './output/glyph_bert-base-chinese' \
  --mode 'glyph' \
  --save_path 'results/retrieval_glyph'

CUDA_VISIBLE_DEVICES=0 python match.py \
  --model './output/meaning_bert-base-chinese' \
  --mode 'meaning' \
  --save_path 'results/retrieval_glyph'

# Generative QA
OPENAI_API_KEY=$KEY python test.py \
  --model 'chatgpt' \
  --mode 'qa'

CUDA_VISIBLE_DEVICES=0 python test.py \
  --model 'chatglm' \
  --mode 'qa'

# Multiple-choice QA
OPENAI_API_KEY=$KEY python test.py \
  --model 'chatgpt' \
  --mode 'mc' \
  --strategy 'random'

CUDA_VISIBLE_DEVICES=0 python test.py \
  --model 'chatglm' \
  --mode 'mc' \
  --strategy 'random'

OPENAI_API_KEY=$KEY python test.py \
  --model 'chatgpt' \
  --mode 'mc' \
  --strategy 'top'

CUDA_VISIBLE_DEVICES=0 python test.py \
  --model 'chatglm' \
  --mode 'mc' \
  --strategy 'top'

- Dependencies

  toolkits:

  ```
  transformers
  sacrebleu
  sentence_transformers
  ```

  data:

  `word.json` which should have been in `src_data` directory comes from [chinese-xinhua](https://github.com/pwxcoo/chinese-xinhua/blob/master/data/word.json).

- Structure

  - Data: The `txt` files in `data` directory(but not in `src_data` directory) are the same as the `csv` files in `CC-Riddle` and `dict` directories.

  - Code

    - Generation (in `generation` directory)

      `create_input.py`: construct train/valid/test data.

      `run_riddle-generation.py`: finetune the pretrained model.

      `generate.py`: validate and test the finetuned model by calculating the `BLEU` score.

      `utils.py`: read and process the files.

    - QA (in `QA` directory)

      `create_train_data.py`: construct train/valid data.

      `training.py`: finetune the pretrained model.

      `riddle_qa.py`: test the finetuned model.
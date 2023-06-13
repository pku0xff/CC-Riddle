# CC-Riddle

CC-Riddle: A Question Answering Dataset of Chinese Character Riddles

- Dependencies

  ```
  transformers
  sacrebleu
  sentence_transformers
  openai
  ```

- Usage

  Before you start, unzip `word.zip` in `data/src_data`.
  
  To run riddle generation experiments, run the following command:
  ```
    cd generation
    bash run.sh
  ```
  
  To run QA experiments, run the following command:
  ```
    cd QA
    bash run.sh
  ```


This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) International License.
We allow the dataset to be used for non-commercial purposes only.
```

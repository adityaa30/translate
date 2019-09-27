# Translate

A neural machine translation system is a neural network that directly models the conditional probability *p(y|x)* of translating a source sentence, *x<sub>1</sub>, . . . , x<sub>n</sub>* to a target sentence *y<sub>1</sub>, . . . , y<sub>m</sub>*.

A basic form of NMT consists of two components:

1. An **encoder** which computes a representation's for each source sentence.
2. A **decoder** which generates one target word at a time.

## Prerequisites

- [python3](https://www.python.org/downloads/release/python-369/) - version 3.6.9
- [pip](https://pip.pypa.io/en/stable/installing/) - version > 19.0
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)

## Installation

1. Clone the github repository - `git clone https://github.com/adityaa30/translate`
2. Go to project directory - `cd translate`
3. Set up the environment :
    - Create virtual environment files - `virtualenv venv` or `python -m venv venv`
    - Activate virtual environment - `source venv/bin/activate` (for Linux) or `venv\Scripts\activate` (for Windows)
4. Install dependencies - `pip install -r requirements.txt`

## Usage

> $ python translate/main.py --help
```
usage: main.py [-h] [-r REVERSE_DATA] [-d DATASET_PATH] [-l LOG_FILE] -c
               CHECKPOINT_DIR
               {train,evaluate} ...

positional arguments:
  {train,evaluate}

optional arguments:
  -h, --help            show this help message and exit
  -r REVERSE_DATA, --reverse-data REVERSE_DATA
                        True => Use 1st columnt in dataset file as target and
                        2nd as input while training else vice versa.
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path of the directory containing the dataset.
                        Downloads English-Spanish dataset by default.
  -l LOG_FILE, --log-file LOG_FILE
                        Path to the .log file. By default logs will be added
                        to `app.log`
  -c CHECKPOINT_DIR, --checkpoint-dir CHECKPOINT_DIR
                        Loads the checkpoint from the given directory if
                        present. Saves a checkpoint at the end of every 2
```

> $ python translate/main.py train --help
```
usage: main.py train [-h] [-b BATCH_SIZE] [-D DATASET_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size to use while training
  -D DATASET_SIZE, --dataset-size DATASET_SIZE
                        Dataset size to use while training
```

> $ python translate/main.py evaluate --help
```
usage: main.py evaluate [-h] -s SENTENCE

optional arguments:
  -h, --help            show this help message and exit
  -s SENTENCE, --sentence SENTENCE
                        Sentence string to be translated
```

## References

- Bahdanau, D., Cho, K. and Bengio, Y., 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
- Luong, M.T., Pham, H. and Manning, C.D., 2015. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
- [Neural Machine Translation with Attention](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention)

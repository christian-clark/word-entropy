# Word Entropy
Code for calculating the entropy measures used in the paper "How Well Does First-Token Entropy Approximate Word Entropy as a Psycholinguistic Predictor?" by Clark et al. (2025).

## First-token entropy calculation

In the following examples, `<CORPUS>` should contain one sentence per line and separate articles using the delimiter `!ARTICLE`.

Shannon entropy:
```
python3 first_token_entropy.py <CORPUS> gpt2 > <CORPUS>.entropy
```

Rényi entropy with alpha=1/2:
```
python3 first_token_entropy.py <CORPUS> gpt2 -a 0.5 > <CORPUS>.entropy
```

Other command-line arguments:
* `--unrestricted`: calculate first-token entropy over the full token vocabulary T rather than the subset of space-intial tokens T_B
* `--gpu`: use GPU

## Monte Carlo word entropy

Shannon entropy with 64 samples per word:
```
python3 mc_word_entropy.py <CORPUS> gpt2 > <CORPUS>.entropy -s 64 > <CORPUS>.entropy
```

Rényi entropy with 64 samples per word:
```
python3 mc_word_entropy.py <CORPUS> gpt2 > <CORPUS>.entropy -a 0.5 -s 64 > <CORPUS>.entropy
```

Other command-line arguments:
* `--samplesPerBatch`: batch size for word sampling
* `--maxIter`: maximum number of subword tokens in a word
* `--contextSize`: length of context window (if unspecified, the full context window will be used)
* `--seed`: random seed
* `--gpu`: use GPU

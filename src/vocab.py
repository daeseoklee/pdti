from paths import paths
from pathlib import Path

__author__ = 'Daeseok Lee'

        

_PR_VOCAB = []
with open(paths.pr_vocab, 'r') as reader:
    for line in reader.readlines():
        _PR_VOCAB.append(line.strip())

PR_VOCAB = {x : i for i, x in enumerate(_PR_VOCAB)}
PR_ALPH_RNG = (_PR_VOCAB.index('L'), _PR_VOCAB.index('X'))

_SM_VOCAB = [] 
with open(paths.sm_vocab, 'r') as reader:
    for line in reader.readlines():
        _SM_VOCAB.append(line.strip())
SM_VOCAB = {x : i for i, x in enumerate(_SM_VOCAB)}
SM_ALPH_RNG = (_SM_VOCAB.index('('), _SM_VOCAB.index('y'))


if __name__ == '__main__':
    print(PR_VOCAB)
    print(SM_VOCAB)
    





 



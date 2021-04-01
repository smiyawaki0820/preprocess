# Sentence Piece
* https://github.com/rsennrich/subword-nmt
* https://colab.research.google.com/drive/1AGnN3THAsjNhlsqILswuGUsBAnbTGusv

## Set up
```bash
$ pip install pip-tools
$ pip-compile requirements.in
$ pip-sync
% bash scripts/downloads.sh
```

## Preprocess
```python
% python src/preprocess.py
```


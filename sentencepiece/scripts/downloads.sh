#!/usr/bin/bash
set -e

mkdir datasets
wget -nc https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt -P datasets

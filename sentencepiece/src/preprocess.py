import os
import io
import pickle
import argparse
from pprint import pprint
from typing import Union, List

import sentencepiece as spm


class MySentencePiece(object):
    # https://colab.research.google.com/drive/1AGnN3THAsjNhlsqILswuGUsBAnbTGusv#scrollTo=07FMNoCmglil
    # https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt
    # 作成ファイル ... prefix.model, prefix.vocab (debug)
    def __init__(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(description='SentencePiece')
        parser = self.spm_parser(parser)
        self.args = parser.parse_args()
        args = self.args
        pprint(args.__dict__)

        assert bool(args.sp_fi_text) ^ bool(args.sp_fi_model), '"args.fi_text" か "args.fi_model" のどちらかを選択して下さい'

        if args.sp_fi_text:
            self.model_proto = io.BytesIO()
            spm.SentencePieceTrainer.train(
                input=args.sp_fi_text, 
                # sentence_iterator=iter(text_list) # input ではなく、iter を指定しても良い
                model_writer=self.model_proto,  # model_prefix=args.sp_prefix
                model_type=args.sp_model_type,
                normalization_rule_name=args.sp_normalize, # normalization_rule_tsv で独自ルールを使用することも可能
                vocab_size=args.sp_vocab_size,
                pad_id=1, pad_piece='[PAD]',
                unk_id=2, unk_piece='[UNK]',
                bos_id=3, bos_piece='[BOS]',
                eos_id=-1, eos_piece='[EOS]',
                user_defined_symbols=['[SEP]', '[CLS]'],
                # control_symbols=['[SEP]', '[CLS]'],   # 入力中に出現すると想定されない場合
                byte_fallback=args.sp_fallback,
                split_by_whitespace=args.sp_space_split,
            )
            self.processor = spm.SentencePieceProcessor(model_proto=self.model_proto.getvalue())
        elif args.sp_fi_model:
            self.load(args.sp_fi_model)

        self.write(f'{args.sp_prefix}.model', dest=args.sp_dest)
        
    def spm_parser(self, parser):
        _spm = parser.add_argument_group('Group of Sentence Piece')
        _spm.add_argument('--sp_fi_text', default='datasets/botchan.txt', type=str, help='input file')
        _spm.add_argument('--sp_fi_model', default=None, type=str, help='input model')
        _spm.add_argument('--sp_dest', default='models', type=str, help='dest of model')
        _spm.add_argument('--sp_prefix', default='spm_botchan', type=str, help='model prefix')
        _spm.add_argument('--sp_vocab_size', default=2000, type=int)
        _spm.add_argument('--sp_fallback', action='store_true', help='未知文字を utf8 文字に分解するか')
        _spm.add_argument('--sp_space_split', action='store_true', help='空白をトークンの区切りとするか')
        _spm.add_argument('--sp_model_type', default='bpe', choices=['bpe', 'word', 'char'])
        _spm.add_argument('--sp_normalize', default='nmt_nfkc', 
            choices=['nmt_nfkc', 'nfkc', 'nmt_nfkc_cf', 'nfkc_cf', 'identity'],
            help='[Unicode NFKC + 空白まわりの独自ルール, Unicode NFKC, nmt_nfkc + 小文字化, nfkc + 小文字化, 正規化なし]'
        )
        return parser

    def write(self, fo_model:str, dest='models'):
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, fo_model), 'wb') as fo:
            pickle.dump(self.processor, fo)
    
    def load(self, fi_model:str):
        with open(fi_model, 'rb') as fi:
            self.processor = pickle.load(fi)

    def encode(self, text:Union[List[str], str], out_type:type=int, bos=True, eos=True):
        # enable_sampling(bool):サブワード正則化, alpha(float):分布の偏り, nbest_size(int):探索空間を限定すると最適解に近い分割に制限しやすくする(unigram)
        # self.processor.set_vocabulary(vocab_list)
        # self.reset_vocabulary()
        return self.processor.encode(text, out_type=out_type, add_bos=bos, add_eos=eos)

    def decode(self, ids:Union[List[List[int]], List[int], List[str]]) -> Union[List[str], str]:
        return self.processor.decode(ids)

    def id_to_piece(self, ids:Union[List[int], int]) -> Union[List[str], str]:
        return self.processor.id_to_piece(ids)
    
    def piece_to_id(self, piece:Union[List[str], str]) -> Union[List[int], int]:
        return self.processor.piece_to_id(piece)


def run():
    sp = MySentencePiece()
    print(sp.encode('I am a university student.'))
    

if __name__ == '__main__':
    run()

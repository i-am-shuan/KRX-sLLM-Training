#scripts/tokenization_qwen2.py
from transformers.tokenization_utils import PreTrainedTokenizer
import json
import os
import unicodedata
from typing import Optional, List, Dict, Tuple
import regex as re

class Qwen2Tokenizer(PreTrainedTokenizer):
    """Qwen2 토크나이저 구현"""
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: Optional[str] = "<|im_start|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        clean_up_tokenization_spaces: bool = False,
        **kwargs
    ):
        # Basic encoder initialization
        self.encoder = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|user|>": 3,
            "<|assistant|>": 4,
        }
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Call parent class constructor
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        """Returns the full vocabulary"""
        return self.encoder

    def _tokenize(self, text: str) -> List[str]:
        """Basic character-level tokenization"""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to its ID"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an ID to its token"""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts tokens back to a string"""
        return "".join(tokens)

    def prepare_for_tokenization(self, text: str, **kwargs) -> Tuple[str, Dict]:
        """Performs any pre-tokenization steps"""
        text = unicodedata.normalize("NFC", text)
        return text, kwargs

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to files"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Builds model inputs by concatenating and adding special tokens"""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

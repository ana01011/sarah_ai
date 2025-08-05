"""
Advanced Tokenization and Text Preprocessing Service
Handles text tokenization, encoding, and preprocessing for the neural network
"""
import re
import json
import pickle
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
import unicodedata
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class AdvancedTokenizer:
    """
    Advanced tokenizer with custom vocabulary and preprocessing capabilities
    """
    
    def __init__(self, vocab_size: int = 50000, min_frequency: int = 2,
                 special_tokens: Optional[Dict[str, str]] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = special_tokens or {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "mask_token": "<mask>"
        }
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_frequencies: Counter = Counter()
        
        # Text preprocessing components
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()
        
        # Regex patterns for text cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        logger.info("Initialized AdvancedTokenizer")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text using Unicode normalization"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs, emails, mentions, hashtags
        text = self.url_pattern.sub('<url>', text)
        text = self.email_pattern.sub('<email>', text)
        text = self.mention_pattern.sub('<mention>', text)
        text = self.hashtag_pattern.sub('<hashtag>', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text: str, remove_stopwords: bool = False,
                       lemmatize: bool = False) -> str:
        """Advanced text preprocessing"""
        # Normalize text
        text = self.normalize_text(text)
        
        # Tokenize into words
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def build_vocabulary(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """Build vocabulary from training texts"""
        logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Count token frequencies
        for text in texts:
            preprocessed = self.preprocess_text(text)
            tokens = self.tokenize(preprocessed)
            self.token_frequencies.update(tokens)
        
        # Add special tokens first
        for i, (name, token) in enumerate(self.special_tokens.items()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Add most frequent tokens up to vocab_size
        most_common = self.token_frequencies.most_common(
            self.vocab_size - len(self.special_tokens)
        )
        
        current_id = len(self.special_tokens)
        for token, freq in most_common:
            if freq >= self.min_frequency and token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        logger.info(f"Built vocabulary with {len(self.token_to_id)} tokens")
        
        # Save vocabulary if path provided
        if save_path:
            self.save_vocabulary(save_path)
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization (can be overridden for more advanced methods)"""
        # Simple whitespace tokenization with punctuation handling
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None,
               add_special_tokens: bool = True, padding: bool = False,
               truncation: bool = False) -> Dict[str, Any]:
        """Encode text to token IDs"""
        # Preprocess and tokenize
        preprocessed = self.preprocess_text(text)
        tokens = self.tokenize(preprocessed)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.special_tokens["bos_token"]] + tokens + [self.special_tokens["eos_token"]]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id[self.special_tokens["unk_token"]])
        
        # Handle truncation
        if truncation and max_length and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep BOS and EOS tokens
                token_ids = ([token_ids[0]] + 
                           token_ids[1:max_length-1] + 
                           [token_ids[-1]])
            else:
                token_ids = token_ids[:max_length]
        
        # Handle padding
        attention_mask = [1] * len(token_ids)
        if padding and max_length and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            pad_id = self.token_to_id[self.special_tokens["pad_token"]]
            token_ids.extend([pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "tokens": tokens
        }
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    add_special_tokens: bool = True, padding: bool = True,
                    truncation: bool = True) -> Dict[str, List[Any]]:
        """Encode batch of texts"""
        batch_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "tokens": []
        }
        
        for text in texts:
            encoding = self.encode(
                text, max_length=max_length,
                add_special_tokens=add_special_tokens,
                padding=padding, truncation=truncation
            )
            
            batch_encodings["input_ids"].append(encoding["input_ids"])
            batch_encodings["attention_mask"].append(encoding["attention_mask"])
            batch_encodings["tokens"].append(encoding["tokens"])
        
        return batch_encodings
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                tokens.append(token)
        
        # Join tokens (basic detokenization)
        text = ' '.join(tokens)
        
        # Basic punctuation handling
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        text = re.sub(r' \'', r'\'', text)
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)
    
    def get_special_token_id(self, token_name: str) -> int:
        """Get ID of special token"""
        if token_name in self.special_tokens:
            return self.token_to_id[self.special_tokens[token_name]]
        raise ValueError(f"Unknown special token: {token_name}")
    
    def save_vocabulary(self, save_path: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
            "token_frequencies": dict(self.token_frequencies),
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vocabulary to {save_path}")
    
    def load_vocabulary(self, load_path: str) -> None:
        """Load vocabulary from file"""
        with open(load_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.special_tokens = vocab_data["special_tokens"]
        self.token_frequencies = Counter(vocab_data["token_frequencies"])
        self.vocab_size = vocab_data["vocab_size"]
        self.min_frequency = vocab_data["min_frequency"]
        
        logger.info(f"Loaded vocabulary from {load_path}")


class HuggingFaceTokenizerWrapper:
    """
    Wrapper for HuggingFace tokenizers with additional preprocessing
    """
    
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized HuggingFace tokenizer: {model_name}")
    
    def encode(self, text: str, max_length: Optional[int] = None,
               padding: bool = False, truncation: bool = False) -> Dict[str, Any]:
        """Encode text using HuggingFace tokenizer"""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze().tolist(),
            "attention_mask": encoding["attention_mask"].squeeze().tolist()
        }
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Encode batch of texts"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size


class TokenizerService:
    """
    Service class for managing tokenization in the application
    """
    
    def __init__(self, tokenizer_type: str = "custom", **kwargs):
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "custom":
            self.tokenizer = AdvancedTokenizer(**kwargs)
        elif tokenizer_type == "huggingface":
            model_name = kwargs.get("model_name", "gpt2")
            self.tokenizer = HuggingFaceTokenizerWrapper(model_name)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        logger.info(f"Initialized TokenizerService with {tokenizer_type} tokenizer")
    
    def prepare_training_data(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Prepare data for training"""
        if self.tokenizer_type == "custom":
            batch_encoding = self.tokenizer.encode_batch(
                texts, max_length=max_length, padding=True, truncation=True
            )
            return {
                "input_ids": torch.tensor(batch_encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(batch_encoding["attention_mask"], dtype=torch.long)
            }
        else:
            return self.tokenizer.encode_batch(texts, max_length=max_length)
    
    def prepare_inference_data(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Prepare single text for inference"""
        encoding = self.tokenizer.encode(text, max_length=max_length, padding=True, truncation=True)
        
        if self.tokenizer_type == "custom":
            return {
                "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long)
            }
        else:
            return {
                "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long)
            }
    
    def decode_response(self, token_ids: torch.Tensor) -> str:
        """Decode model output to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze().tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.get_vocab_size()
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        if self.tokenizer_type == "custom":
            return {
                name: self.tokenizer.get_special_token_id(name)
                for name in self.tokenizer.special_tokens.keys()
            }
        else:
            return {
                "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.tokenizer.eos_token_id,
                "bos_token_id": getattr(self.tokenizer.tokenizer, 'bos_token_id', None),
                "unk_token_id": getattr(self.tokenizer.tokenizer, 'unk_token_id', None)
            }
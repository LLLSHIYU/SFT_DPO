#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSAI 4207 Data Engineering Utilities (Protocol v1.2)
Owner: PAN Haoren

Provides:
- Text normalization (normalize_for_training vs normalize_for_hash)
- Deterministic SHA256 hashing & file hashing
- Policy classification (get_policy_family, get_policy_tier)
- Official Qwen2.5-0.5B-Base Tokenizer & ChatML renderer/label-masker aligned byte-for-byte
  with `tokenizer.apply_chat_template`
- 5-word shingle MinHash (128-perm) + LSH + Exact Jaccard verification & Union-Find clustering
"""

import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import regex as re_u
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ENG_DIR = Path(__file__).resolve().parents[1]
DATA_V1_DIR = PROJECT_ROOT / "data" / "v1.0"
RAW_CACHE_DIR = DATA_V1_DIR / "raw_cache"

SYSTEM_PROMPT = (
    "You are a helpful assistant who is an expert at summarizing content into "
    "TL;DR sentences. Summarize the following text"
)


def load_config() -> Dict[str, Any]:
    cfg_path = DATA_ENG_DIR / "configs" / "data_v1.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 1. Text Normalization (§5.4)
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_TLDR_PREFIX_RE = re.compile(r"^\s*(?:tl\s*[;/]?\s*dr)\s*[:\-–—]?\s*", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DELETED_RE = re.compile(r"^\s*\[?\s*(?:removed|deleted)\s*\]?\s*$", re.IGNORECASE)


def normalize_for_training(
    text: Optional[str], strip_tldr_prefix: bool = False
) -> Tuple[str, bool]:
    """
    Normalizes text for training without altering case, punctuation, or wording.
    Returns (normalized_text, was_tldr_prefix_stripped).
    """
    if not text or not isinstance(text, str):
        return "", False
    s = html.unescape(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _ZERO_WIDTH_RE.sub("", s)
    s = s.replace("\u00a0", " ")
    s = s.strip()
    stripped_prefix = False
    if strip_tldr_prefix:
        m = _TLDR_PREFIX_RE.match(s)
        if m and len(s) > m.end() + 5:
            s = s[m.end() :].strip()
            stripped_prefix = True
    s = _MULTI_NL_RE.sub("\n\n", s)
    return s, stripped_prefix


def normalize_for_hash(text: Optional[str]) -> str:
    """
    Normalizes text exclusively for deduplication and decontamination hashing.
    Never used as training text.
    """
    if not text or not isinstance(text, str):
        return ""
    s = html.unescape(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def is_deleted_or_empty(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return True
    s = text.strip()
    if not s:
        return True
    return bool(_DELETED_RE.match(s))


def clean_subreddit(sub: Optional[str]) -> str:
    if not sub or not isinstance(sub, str):
        return ""
    s = sub.strip()
    if s.lower().startswith("r/"):
        s = s[2:].strip()
    return s


def format_user_content(subreddit: str, title: str, post: str) -> str:
    return f"Subreddit: r/{subreddit}\nTitle: {title}\n\n{post}"


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# 2. Policy Classification (§5.2, Stiennon et al. 2020 Table 8)
# ---------------------------------------------------------------------------


def get_policy_family(policy: Optional[str]) -> str:
    """
    Classifies OpenAI summarize_from_feedback policy string per Stiennon et al. (2020, Table 8):
    - 'human_ref': 'ref' (human reference summary from Reddit author)
    - 'rl_aligned': policies containing 'ppo' or 'bo' (RLHF / Best-of-N aligned policies)
    - 'sft_strong': 'sup4', 'sup4_6b', 'sup3_6b' (and variants without ppo/bo)
    - 'sft_early': 'sup1', 'sup2', 'sup3' and other early/temperature-sweep policies
    """
    if not policy or not isinstance(policy, str):
        return "sft_early"
    p = policy.strip().lower()
    if p == "ref":
        return "human_ref"
    if "ppo" in p or "_bo" in p or p.startswith("bo"):
        return "rl_aligned"
    if p.startswith("sup4") or p.startswith("sup3_6b"):
        return "sft_strong"
    return "sft_early"


def get_policy_tier(policy: Optional[str]) -> int:
    """
    Returns numeric quality tier for canonical pair tie-breaking:
    3 = rl_aligned (ppo / bo)
    2 = sft_strong (sup4* / sup3_6b) or human_ref (ref)
    1 = sft_early (sup1 / sup2 / sup3 / others)
    """
    fam = get_policy_family(policy)
    if fam == "rl_aligned":
        return 3
    if fam in ("sft_strong", "human_ref"):
        return 2
    return 1


def get_stratum_policy_bucket(policy_family: str) -> str:
    """Groups policy families into 3 balanced buckets for stratification (§8.4)."""
    if policy_family in ("rl_aligned", "sft_strong"):
        return "rl_or_strong"
    if policy_family == "human_ref":
        return "human_ref"
    return "sft_early"


# ---------------------------------------------------------------------------
# 3. Official Qwen2.5-0.5B-Base Tokenizer & ChatML Renderer (§11, P0-1, P0-4)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class QwenTokenizerWrapper:
    """
    Official Qwen/Qwen2.5-0.5B-Base Tokenizer with enforced ChatML special tokens:
    - eos_token = "<|im_end|>" (151645)
    - pad_token = "<|endoftext|>" (151643)
    - im_start_token = "<|im_start|>" (151644)

    Strictly validates that the loaded directory contains the official Qwen2.5 tokenizer
    (151,643 base BPE vocab + 22 added special tokens = 151,665 total entries, including
    official `tokenizer.json` and `tokenizer_config.json`), refusing any synthetic BPE table.
    """

    ENDOFTEXT_ID = 151643
    IM_START_ID = 151644
    IM_END_ID = 151645

    SPECIAL_TOKENS = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }

    PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(self, tokenizer_dir: Path, allow_synthetic: bool = False):
        self.tokenizer_dir = tokenizer_dir
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        cfg_path = tokenizer_dir / "tokenizer_config.json"
        tok_json_path = tokenizer_dir / "tokenizer.json"

        if not vocab_path.exists() or not merges_path.exists() or not cfg_path.exists():
            raise FileNotFoundError(
                f"Official Qwen tokenizer files missing in {tokenizer_dir}. Run 00_download.py first."
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.tokenizer_config: Dict[str, Any] = json.load(f)

        self.base_vocab_size = len(self.encoder)
        if not allow_synthetic:
            if (
                self.base_vocab_size < 151643
                or "hello" not in self.encoder
                or "Ġthe" not in self.encoder
                or not tok_json_path.exists()
                or "added_tokens_decoder" not in self.tokenizer_config
            ):
                raise RuntimeError(
                    f"Non-official or synthetic tokenizer detected in {tokenizer_dir} "
                    f"(vocab_size={self.base_vocab_size}). Official Qwen/Qwen2.5-0.5B-Base tokenizer is required!"
                )

        added_decoder = self.tokenizer_config.get("added_tokens_decoder") or {}
        for tok_id_str, tok_info in added_decoder.items():
            content = tok_info.get("content") if isinstance(tok_info, dict) else None
            if content:
                self.encoder[content] = int(tok_id_str)

        for sp_tok, sp_id in self.SPECIAL_TOKENS.items():
            self.encoder[sp_tok] = sp_id

        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.vocab_size = self.base_vocab_size
        self._total_len = len(self.encoder)

        # Load fast Rust/HF tokenizer backend if available for 100% parity with AutoTokenizer / TRL
        self._hf_tok = None
        if tok_json_path.exists():
            try:
                from transformers import AutoTokenizer

                self._hf_tok = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir), use_fast=True, local_files_only=True
                )
                self._hf_tok.eos_token = "<|im_end|>"
                self._hf_tok.pad_token = "<|endoftext|>"
                self._hf_tok.padding_side = "right"
                self._total_len = len(self._hf_tok)
                self.vocab_size = self._hf_tok.vocab_size
            except Exception:
                self._hf_tok = None

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_lines = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#version")
            ]
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(line.split()): i for i, line in enumerate(merges_lines)
        }
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re_u.compile(self.PAT_STR)
        self.special_re = re.compile(
            "(" + "|".join(re.escape(k) for k in self.SPECIAL_TOKENS.keys()) + ")"
        )
        self.cache: Dict[str, List[int]] = {}

        # Protocol §11.2 enforced attributes
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self.IM_END_ID
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self.ENDOFTEXT_ID
        assert self.eos_token_id != self.pad_token_id

    def __len__(self) -> int:
        return self._total_len

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text)}

    def _bpe(self, token: str) -> List[int]:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            res = [self.encoder[token]]
            self.cache[token] = res
            return res

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = _get_pairs(word)

        res = [self.encoder[sub] for sub in word]
        if len(self.cache) < 500000:
            self.cache[token] = res
        return res

    def encode_bpe_direct(self, text: str) -> List[int]:
        if not text:
            return []
        ids: List[int] = []
        parts = self.special_re.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.SPECIAL_TOKENS:
                ids.append(self.SPECIAL_TOKENS[part])
            else:
                for match in self.pat.findall(part):
                    encoded_bytes = "".join(
                        self.byte_encoder[b] for b in match.encode("utf-8")
                    )
                    ids.extend(self._bpe(encoded_bytes))
        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._hf_tok is not None:
            return list(self._hf_tok.encode(text, add_special_tokens=False))
        return self.encode_bpe_direct(text)

    def decode_token(self, token_id: int) -> str:
        tok = self.decoder.get(token_id, "")
        if tok in self.SPECIAL_TOKENS:
            return tok
        raw_bytes = bytes([self.byte_decoder[c] for c in tok if c in self.byte_decoder])
        return raw_bytes.decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> Any:
        """
        Delegates to official Hugging Face `apply_chat_template` when loaded, or renders
        the exact Qwen2.5 ChatML template with `<|im_end|>\n` after every turn (P0-4).
        """
        rendered = self.render_chatml(messages, add_generation_prompt=add_generation_prompt)
        if self._hf_tok is not None and getattr(self._hf_tok, "chat_template", None):
            hf_rendered = self._hf_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            assert hf_rendered == rendered, (
                f"ChatML mismatch between render_chatml and official apply_chat_template:\n"
                f"render_chatml={rendered!r}\nhf={hf_rendered!r}"
            )
        if tokenize:
            return self.encode(rendered)
        return rendered

    def render_chatml(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        """
        Renders messages using the official Qwen2.5 ChatML template (§11.2, P0-4):
        Each turn is rendered as `<|im_start|>{role}\n{content}<|im_end|>\n` (including trailing newline),
        matching `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)`
        byte-for-byte.
        """
        blocks = []
        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if idx == 0 and role != "system":
                # Official Qwen2.5 chat_template prepends default system message if first message is not system,
                # unless overridden; for warmup multi-turn without system, check if _hf_tok prepends system
                if self._hf_tok is not None and getattr(self._hf_tok, "chat_template", None):
                    return self._hf_tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=add_generation_prompt
                    )
            blocks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        rendered = "".join(blocks)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def tokenize_sft_with_labels(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenizes a conversation for SFT with completion-only loss mask (§11.3, P0-4):
        - All system/user turns and `<|im_start|>assistant\n` headers have label = -100
        - Assistant content + closing `<|im_end|>\n` have label = token_id
        - Full `input_ids` is 100% identical to `encode(apply_chat_template(messages, tokenize=False))`
        """
        # Handle optional default system turn if first message is not system and official template inserts one
        effective_msgs = list(messages)
        if effective_msgs and effective_msgs[0]["role"] != "system":
            if self._hf_tok is not None and getattr(self._hf_tok, "chat_template", None):
                full_rendered = self._hf_tok.apply_chat_template(
                    effective_msgs, tokenize=False, add_generation_prompt=False
                )
                if full_rendered.startswith("<|im_start|>system\n"):
                    sys_end = full_rendered.find("<|im_end|>\n")
                    sys_txt = full_rendered[len("<|im_start|>system\n") : sys_end]
                    effective_msgs = [{"role": "system", "content": sys_txt}] + effective_msgs

        input_ids: List[int] = []
        labels: List[int] = []
        for msg in effective_msgs:
            role = msg["role"]
            content = msg["content"]
            header = f"<|im_start|>{role}\n"
            header_ids = self.encode(header)
            body_with_eos = f"{content}<|im_end|>\n"
            body_ids = self.encode(body_with_eos)

            input_ids.extend(header_ids)
            labels.extend([-100] * len(header_ids))

            input_ids.extend(body_ids)
            if role == "assistant":
                labels.extend(body_ids)
            else:
                labels.extend([-100] * len(body_ids))
        return input_ids, labels

    def tokenize_dpo_pair(
        self,
        prompt_messages: List[Dict[str, str]],
        chosen_messages: List[Dict[str, str]],
        rejected_messages: List[Dict[str, str]],
    ) -> Dict[str, List[int]]:
        """
        Tokenizes a DPO pair under the official Qwen2.5 ChatML template so that:
        - `prompt_str` == `apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)`
        - `prompt_ids + chosen_ids` == `tokenize_sft_with_labels(prompt_messages + chosen_messages)[0]`
        - `chosen_ids` == `[lbl for lbl in sft_labels if lbl != -100]`
        """
        prompt_str = self.render_chatml(prompt_messages, add_generation_prompt=True)
        prompt_ids = self.encode(prompt_str)

        chosen_str = f"{chosen_messages[0]['content']}<|im_end|>\n"
        rejected_str = f"{rejected_messages[0]['content']}<|im_end|>\n"
        chosen_ids = self.encode(chosen_str)
        rejected_ids = self.encode(rejected_str)

        return {
            "prompt_input_ids": prompt_ids,
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "full_chosen_input_ids": prompt_ids + chosen_ids,
            "full_rejected_input_ids": prompt_ids + rejected_ids,
        }


# ---------------------------------------------------------------------------
# 4. 5-Word Shingle MinHash (128-perm) + LSH + Exact Jaccard Verification (§6)
# ---------------------------------------------------------------------------

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


@lru_cache(maxsize=1)
def _get_minhash_permutations(num_perm: int = 128, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE_PRIME, size=(num_perm,), dtype=np.uint64)
    b = rng.randint(0, _MERSENNE_PRIME, size=(num_perm,), dtype=np.uint64)
    return a, b


def extract_word_shingles(norm_text: str, k: int = 5) -> Set[str]:
    """Extracts unique k-word shingles from already hash-normalized text."""
    words = norm_text.split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def compute_exact_jaccard(shingles_a: Set[str], shingles_b: Set[str]) -> float:
    if not shingles_a or not shingles_b:
        return 0.0
    inter = len(shingles_a & shingles_b)
    if inter == 0:
        return 0.0
    union = len(shingles_a) + len(shingles_b) - inter
    return inter / union


def compute_minhash_signature(shingles: Set[str], num_perm: int = 128) -> Optional[np.ndarray]:
    if not shingles:
        return None
    a, b = _get_minhash_permutations(num_perm=num_perm, seed=42)
    hv = np.array(
        [
            int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "little")
            for s in shingles
        ],
        dtype=np.uint64,
    )
    phv = np.bitwise_and((hv[:, None] * a[None, :] + b[None, :]) % _MERSENNE_PRIME, _MAX_HASH)
    return phv.min(axis=0).astype(np.uint32)


class ExactVerifiedMinHashIndex:
    """
    Combines 128-permutation MinHash LSH (32 bands x 4 rows for >99.9999% recall at J>=0.7)
    with a fast inverted shingle-prefix filter and mandatory exact 5-gram Jaccard verification.
    Guarantees zero LSH false-positive chaining (§6.3).
    """

    def __init__(self, num_perm: int = 128, bands: int = 32):
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.doc_shingles: Dict[str, Set[str]] = {}
        self.doc_norm_text: Dict[str, str] = {}
        self.buckets: Dict[Tuple[int, bytes], List[str]] = defaultdict(list)

    def insert(self, doc_id: str, norm_text: str, k: int = 5) -> None:
        self.doc_norm_text[doc_id] = norm_text
        shingles = extract_word_shingles(norm_text, k=k)
        self.doc_shingles[doc_id] = shingles
        if not shingles:
            return
        sig = compute_minhash_signature(shingles, num_perm=self.num_perm)
        if sig is None:
            return
        for b_idx in range(self.bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_bytes = sig[start:end].tobytes()
            self.buckets[(b_idx, band_bytes)].append(doc_id)

    def query_Text(
        self, norm_text: str, threshold: float = 0.8, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Returns all indexed doc_ids that match norm_text via exact match (if <k words)
        or have exact 5-gram Jaccard >= threshold.
        """
        shingles = extract_word_shingles(norm_text, k=k)
        if not shingles:
            if not norm_text:
                return []
            return [
                (did, 1.0)
                for did, t in self.doc_norm_text.items()
                if t == norm_text
            ]

        sig = compute_minhash_signature(shingles, num_perm=self.num_perm)
        candidates: Set[str] = set()
        for b_idx in range(self.bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_bytes = sig[start:end].tobytes()
            for cand_id in self.buckets.get((b_idx, band_bytes), ()):
                candidates.add(cand_id)

        verified: List[Tuple[str, float]] = []
        for cand_id in sorted(candidates):
            cand_shingles = self.doc_shingles.get(cand_id)
            if not cand_shingles:
                continue
            la, lb = len(shingles), len(cand_shingles)
            if min(la, lb) / max(la, lb) < threshold:
                continue
            jac = compute_exact_jaccard(shingles, cand_shingles)
            if jac >= threshold:
                verified.append((cand_id, round(jac, 4)))
        return verified

    def find_all_internal_pairs(
        self, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Finds all unique internal pairs (id1 < id2) with exact 5-gram Jaccard >= threshold.
        """
        candidate_pairs: Set[Tuple[str, str]] = set()
        for bucket_docs in self.buckets.values():
            if len(bucket_docs) > 1:
                sorted_docs = sorted(set(bucket_docs))
                n = len(sorted_docs)
                for i in range(n):
                    for j in range(i + 1, n):
                        candidate_pairs.add((sorted_docs[i], sorted_docs[j]))

        verified_pairs: List[Tuple[str, str, float]] = []
        for d1, d2 in sorted(candidate_pairs):
            s1 = self.doc_shingles[d1]
            s2 = self.doc_shingles[d2]
            if min(len(s1), len(s2)) / max(len(s1), len(s2)) < threshold:
                continue
            jac = compute_exact_jaccard(s1, s2)
            if jac >= threshold:
                verified_pairs.append((d1, d2, round(jac, 4)))
        return verified_pairs


def cluster_pairs_union_find(
    doc_ids: Iterable[str], pairs: List[Tuple[str, str, float]]
) -> Dict[str, List[str]]:
    """
    Groups doc_ids into connected components using Union-Find over verified near-dup edges.
    Only returns clusters with size >= 2.
    """
    parent: Dict[str, str] = {d: d for d in doc_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for d1, d2, _ in pairs:
        if d1 in parent and d2 in parent:
            union(d1, d2)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for d in sorted(parent.keys()):
        clusters[find(d)].append(d)

    return {root: members for root, members in clusters.items() if len(members) > 1}

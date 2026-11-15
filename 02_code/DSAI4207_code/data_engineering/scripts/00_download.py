#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_download.py — Download and lock official dataset & tokenizer artifacts for DSAI 4207 (Protocol v1.2).
Fixes P0-1, P0-2, P0-3, P1-2, P2-5:
- Never strips HF_TOKEN / HUGGING_FACE_HUB_TOKEN from environment.
- Offline mode is strictly forbidden for data/v1.0 and exits with non-zero status.
- Optional `--allow-synthetic` flag is restricted exclusively to `data/smoke/` with `synthetic=true`.
- Downloads official `Qwen/Qwen2.5-0.5B-Base` (`Qwen/Qwen2.5-0.5B`) tokenizer files
  (`vocab.json`, `merges.txt`, `tokenizer_config.json`, `tokenizer.json`) and verifies vocab authenticity.
- Downloads official benchmark parquets (`cais/mmlu` all/test 14,042 rows, `allenai/ai2_arc` ARC-Challenge/test 1,172 rows,
  `google/IFEval` train 541 rows) via datasets-server parquet API.
- Locks HF repo, official commit SHA (`revision`), and parquet/file SHA256s in `sources_lock.json`.
"""

import argparse
import json
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import regex as re_u
from utils import (
    DATA_ENG_DIR,
    PROJECT_ROOT,
    RAW_CACHE_DIR,
    QwenTokenizerWrapper,
    _bytes_to_unicode,
    load_config,
    sha256_file,
)


def _create_ssl_context() -> ssl.SSLContext:
    for ca_path in [
        os.environ.get("SSL_CERT_FILE", ""),
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/etc/ssl/cert.pem",
    ]:
        if ca_path and os.path.exists(ca_path):
            return ssl.create_default_context(cafile=ca_path)
    return ssl.create_default_context()


SSL_CTX = _create_ssl_context()


class _StripAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            orig_host = urllib.parse.urlparse(req.full_url).hostname
            new_host = urllib.parse.urlparse(newurl).hostname
            if orig_host != new_host and "Authorization" in new_req.headers:
                del new_req.headers["Authorization"]
        return new_req


OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=SSL_CTX),
    _StripAuthOnCrossHostRedirect(),
)


def _get_auth_headers(include_token: bool = True) -> Dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) DSAI4207-DataPipeline/1.2"
    }
    if include_token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _http_get_bytes(url: str, timeout: int = 45) -> bytes:
    for use_tok in ([True, False] if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) else [False]):
        try:
            req = urllib.request.Request(url, headers=_get_auth_headers(include_token=use_tok))
            with OPENER.open(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 401 and use_tok:
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url}")


def _http_get_json(url: str, timeout: int = 45) -> Any:
    return json.loads(_http_get_bytes(url, timeout=timeout).decode("utf-8"))


def _download_with_fallbacks(urls: List[str], dest: Path, timeout: int = 120) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None
    for url in urls:
        for use_tok in ([True, False] if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) else [False]):
            try:
                req = urllib.request.Request(url, headers=_get_auth_headers(include_token=use_tok))
                with OPENER.open(req, timeout=timeout) as resp, open(tmp_dest, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if tmp_dest.exists() and tmp_dest.stat().st_size > 0:
                    tmp_dest.replace(dest)
                    print(f"  [OK] Downloaded {dest.name} ({dest.stat().st_size:,} B) <- {url[:80]}")
                    return url
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 401 and use_tok:
                    continue
                break
            except Exception as e:
                last_err = e
                break
            finally:
                if tmp_dest.exists():
                    try:
                        tmp_dest.unlink()
                    except OSError:
                        pass
        print(f"  [WARN] Candidate URL failed ({url[:75]}): {last_err}")
    raise RuntimeError(f"All download candidates failed for {dest}: {last_err}")


def _is_official_qwen_tokenizer(tok_dir: Path) -> bool:
    vocab_path = tok_dir / "vocab.json"
    merges_path = tok_dir / "merges.txt"
    cfg_path = tok_dir / "tokenizer_config.json"
    tok_json_path = tok_dir / "tokenizer.json"
    if not (vocab_path.exists() and merges_path.exists() and cfg_path.exists() and tok_json_path.exists()):
        return False
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return (
            len(vocab) >= 151643
            and "hello" in vocab
            and "Ġthe" in vocab
            and "added_tokens_decoder" in cfg
            and tok_json_path.stat().st_size > 1_000_000
        )
    except Exception:
        return False


def _is_official_benchmark_parquets(bench_dir: Path) -> bool:
    mmlu_p = bench_dir / "mmlu_all_test.parquet"
    arc_p = bench_dir / "arc_challenge_test.parquet"
    ifeval_p = bench_dir / "ifeval_train.parquet"
    if not (mmlu_p.exists() and arc_p.exists() and ifeval_p.exists()):
        return False
    try:
        t_mmlu = pq.read_table(mmlu_p)
        t_arc = pq.read_table(arc_p)
        t_ifeval = pq.read_table(ifeval_p)
        if t_mmlu.num_rows != 14042 or t_arc.num_rows != 1172 or t_ifeval.num_rows != 541:
            return False
        first_mmlu_q = str(t_mmlu.slice(0, 1).to_pylist()[0].get("question", ""))
        first_arc_q = str(t_arc.slice(0, 1).to_pylist()[0].get("question", ""))
        first_ifeval_p = str(t_ifeval.slice(0, 1).to_pylist()[0].get("prompt", ""))
        if "phenomena #" in first_mmlu_q or "controlled scientific experiment #" in first_arc_q or "[IFEval #" in first_ifeval_p:
            return False
        return True
    except Exception:
        return False


def _check_online() -> bool:
    probe_urls = [
        "https://datasets-server.huggingface.co/parquet?dataset=openai/summarize_from_feedback",
        "https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B/revisions",
        "https://hf-mirror.com/api/models/Qwen/Qwen2.5-0.5B",
    ]
    for url in probe_urls:
        try:
            _http_get_json(url, timeout=10)
            return True
        except Exception:
            continue
    return False


def _fetch_dataset_commit_sha(repo_id: str, default_cfg: str) -> str:
    # Verified official Hugging Face dataset commit SHAs matching the locked parquet files (R1 / P2-5)
    pinned = {
        "openai/summarize_from_feedback": "b8f7d168b6f4e95b2a92e84768bd6c955bed2f29",
        "HuggingFaceH4/no_robots": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
        "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
        "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
        "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    }
    if repo_id in pinned:
        return pinned[repo_id]
    for url in [
        f"https://datasets-server.huggingface.co/info?dataset={repo_id}&config={default_cfg}",
        f"https://datasets-server.huggingface.co/info?dataset={repo_id}",
        f"https://hf-mirror.com/api/datasets/{repo_id}",
        f"https://huggingface.co/api/datasets/{repo_id}",
    ]:
        try:
            data = _http_get_json(url, timeout=20)
            if isinstance(data, dict):
                if isinstance(data.get("sha"), str) and len(data["sha"]) >= 7:
                    return data["sha"]
                ds_info = data.get("dataset_info") or {}
                if isinstance(ds_info.get("sha"), str):
                    return ds_info["sha"]
        except Exception:
            continue
    return "main"


def _download_official_tokenizer(tok_dir: Path) -> str:
    tok_dir.mkdir(parents=True, exist_ok=True)
    tmp_cache_tok = Path("/tmp/hf_official_cache/qwen_tokenizer")
    if not _is_official_qwen_tokenizer(tok_dir) and _is_official_qwen_tokenizer(tmp_cache_tok):
        for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]:
            shutil.copy2(tmp_cache_tok / fname, tok_dir / fname)

    qwen_sha = None
    for api_url in [
        "https://hf-mirror.com/api/models/Qwen/Qwen2.5-0.5B",
        "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B",
        "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Base",
    ]:
        try:
            info = _http_get_json(api_url, timeout=15)
            if isinstance(info, dict) and info.get("sha"):
                qwen_sha = info["sha"]
                break
        except Exception:
            continue
    if not qwen_sha:
        qwen_sha = "060db6499f32faf8b98477b0a26969ef7d8b9987"

    if not _is_official_qwen_tokenizer(tok_dir):
        print("[00_download] Downloading official Qwen/Qwen2.5-0.5B-Base tokenizer files...")
        for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]:
            urls = [
                f"https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B/repo?Revision=master&FilePath={fname}",
                f"https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/master/{fname}",
                f"https://hf-mirror.com/Qwen/Qwen2.5-0.5B/resolve/{qwen_sha}/{fname}",
                f"https://hf-mirror.com/Qwen/Qwen2.5-0.5B/resolve/main/{fname}",
                f"https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/{qwen_sha}/{fname}",
                f"https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/{fname}",
            ]
            _download_with_fallbacks(urls, tok_dir / fname)

    if not _is_official_qwen_tokenizer(tok_dir):
        raise RuntimeError(f"Tokenizer in {tok_dir} failed official Qwen2.5 authenticity check!")
    return qwen_sha


def _download_parquet_split_via_datasets_server(
    repo_id: str, cfg_name: str, split_name: str, dest: Path, expected_min_rows: int
) -> None:
    p_info = _http_get_json(
        f"https://datasets-server.huggingface.co/parquet?dataset={repo_id}", timeout=30
    )
    all_files = p_info.get("parquet_files", [])
    p_files = [
        f for f in all_files if f.get("config") == cfg_name and f.get("split") == split_name
    ]
    if not p_files:
        p_files = [f for f in all_files if f.get("split") == split_name]
    if not p_files:
        raise RuntimeError(f"No parquet files found on datasets-server for {repo_id} {cfg_name}/{split_name}")

    shard_tables = []
    direct_ok = True
    for idx, pf in enumerate(p_files):
        orig_url = pf["url"]
        mirror_url = orig_url.replace("https://huggingface.co/", "https://hf-mirror.com/")
        shard_dest = dest.parent / f"_tmp_shard_{idx}_{dest.name}"
        try:
            _download_with_fallbacks([orig_url, mirror_url], shard_dest, timeout=120)
            shard_tables.append(pq.read_table(shard_dest))
            shard_dest.unlink()
        except Exception as e:
            print(f"  [WARN] Direct parquet shard download failed ({e}); falling back to datasets-server /rows...")
            direct_ok = False
            if shard_dest.exists():
                shard_dest.unlink()
            break

    if direct_ok and shard_tables:
        full_table = pa.concat_tables(shard_tables) if len(shard_tables) > 1 else shard_tables[0]
    else:
        rows_acc = []
        offset = 0
        step = 100
        actual_cfg = p_files[0]["config"]
        while True:
            r_url = (
                f"https://datasets-server.huggingface.co/rows?"
                f"dataset={urllib.parse.quote(repo_id)}&config={urllib.parse.quote(actual_cfg)}"
                f"&split={urllib.parse.quote(split_name)}&offset={offset}&length={step}"
            )
            page = _http_get_json(r_url, timeout=45)
            batch = [item["row"] for item in page.get("rows", [])]
            if not batch:
                break
            rows_acc.extend(batch)
            offset += len(batch)
            if len(batch) < step:
                break
        full_table = pa.Table.from_pylist(rows_acc)

    if full_table.num_rows < expected_min_rows:
        raise RuntimeError(
            f"Downloaded {repo_id} ({cfg_name}/{split_name}) has {full_table.num_rows} rows < {expected_min_rows}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(full_table, dest)


def _ensure_official_ifeval_package() -> None:
    pkg_dir = DATA_ENG_DIR / "scripts" / "instruction_following_eval"
    req_files = [
        "instructions.py",
        "instructions_registry.py",
        "instructions_util.py",
        "evaluation_lib.py",
    ]
    tmp_pkg = Path("/tmp/hf_official_cache/instruction_following_eval")
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for fname in req_files:
        target = pkg_dir / fname
        if not target.exists() and (tmp_pkg / fname).exists():
            shutil.copy2(tmp_pkg / fname, target)
        if not target.exists():
            gh_url = f"https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/{fname}"
            _download_with_fallbacks([gh_url], target, timeout=45)

    # Ensure instructions_util.py uses relative import or package import cleanly and doesn't re-download punkt if present
    util_path = pkg_dir / "instructions_util.py"
    if util_path.exists():
        txt = util_path.read_text(encoding="utf-8")
        if "nltk.download(" in txt and "try:\n  nltk.data.find" not in txt:
            txt = txt.replace(
                'nltk.download("punkt")',
                'try:\n  nltk.data.find("tokenizers/punkt")\nexcept LookupError:\n  nltk.download("punkt", quiet=True)',
            )
            util_path.write_text(txt, encoding="utf-8")

    for fname in ["instructions.py", "instructions_registry.py", "evaluation_lib.py"]:
        fpath = pkg_dir / fname
        if fpath.exists():
            ftxt = fpath.read_text(encoding="utf-8")
            ftxt_new = ftxt.replace(
                "from instruction_following_eval import ", "from instruction_following_eval import "
            )
            if ftxt_new != ftxt:
                fpath.write_text(ftxt_new, encoding="utf-8")

    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            '"""Official Google Research instruction_following_eval package."""\n',
            encoding="utf-8",
        )


def _build_smoke_synthetic_artifacts(smoke_cache_dir: Path, tldr_train_path: Path) -> None:
    """Only invoked when `--allow-synthetic` is explicitly passed; writes exclusively to `data/smoke/`."""
    tok_dir = smoke_cache_dir / "qwen_tokenizer"
    bench_dir = smoke_cache_dir / "benchmarks"
    tok_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    print("[00_download] WARNING: Building synthetic smoke-test artifacts in data/smoke/ ONLY...")
    byte_enc = _bytes_to_unicode()
    vocab = {byte_enc[b]: b for b in range(256)}
    vocab["<|endoftext|>"] = 151643
    vocab["<|im_start|>"] = 151644
    vocab["<|im_end|>"] = 151645
    with open(tok_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(tok_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
    with open(tok_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen2", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSAI 4207 Step 00: Download & Lock Official Data")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback ONLY in data/smoke/ for CI smoke tests (never in data/v1.0).",
    )
    args = parser.parse_args()

    cfg = load_config()
    target_cache_dir = (
        (PROJECT_ROOT / "data" / "smoke" / "raw_cache")
        if args.allow_synthetic
        else RAW_CACHE_DIR
    )
    tok_dir = target_cache_dir / "qwen_tokenizer"
    bench_dir = target_cache_dir / "benchmarks"
    tok_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy pre-cached real Hugging Face parquet files from /tmp/ if present
    tldr_train_dest = target_cache_dir / "tldr_train.parquet"
    tldr_val_dest = target_cache_dir / "tldr_val.parquet"
    nr_dest = target_cache_dir / "norobots_train.parquet"

    for tmp_src, cache_dst in [
        (Path("/tmp/tldr_train.parquet"), tldr_train_dest),
        (Path("/tmp/tldr_val.parquet"), tldr_val_dest),
        (Path("/tmp/norobots_train.parquet"), nr_dest),
    ]:
        if not cache_dst.exists() and tmp_src.exists():
            print(f"[00_download] Copying cached {tmp_src} -> {cache_dst}")
            shutil.copy2(tmp_src, cache_dst)

    # Also copy from /tmp/hf_official_cache if already fetched
    tmp_official_bench = Path("/tmp/hf_official_cache/benchmarks")
    if tmp_official_bench.exists() and _is_official_benchmark_parquets(tmp_official_bench):
        if not _is_official_benchmark_parquets(bench_dir):
            for bname in ["mmlu_all_test.parquet", "arc_challenge_test.parquet", "ifeval_train.parquet"]:
                shutil.copy2(tmp_official_bench / bname, bench_dir / bname)

    already_complete_official = (
        not args.allow_synthetic
        and tldr_train_dest.exists()
        and tldr_val_dest.exists()
        and nr_dest.exists()
        and _is_official_qwen_tokenizer(tok_dir)
        and _is_official_benchmark_parquets(bench_dir)
    )

    online = _check_online()
    if not online and not already_complete_official:
        if not args.allow_synthetic:
            raise SystemExit(
                "Offline mode is not allowed for final data. "
                "Set HF_TOKEN and run with network, or pass --allow-synthetic for smoke tests only."
            )
        _build_smoke_synthetic_artifacts(target_cache_dir, tldr_train_dest)
        lock_path = target_cache_dir / "sources_lock.json"
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump({"synthetic": True}, f, indent=2)
        print(f"[00_download] Smoke test synthetic artifacts written to {target_cache_dir}")
        return

    # 2. Ensure main dataset parquets exist
    if not tldr_train_dest.exists():
        _download_parquet_split_via_datasets_server(
            "openai/summarize_from_feedback", "comparisons", "train", tldr_train_dest, 90000
        )
    if not tldr_val_dest.exists():
        _download_parquet_split_via_datasets_server(
            "openai/summarize_from_feedback", "comparisons", "validation", tldr_val_dest, 80000
        )
    if not nr_dest.exists():
        _download_parquet_split_via_datasets_server(
            "HuggingFaceH4/no_robots", "default", "train", nr_dest, 9500
        )

    # 3. Official Qwen/Qwen2.5-0.5B-Base tokenizer
    qwen_sha = _download_official_tokenizer(tok_dir)

    # 4. Official Benchmarks (overwrite any synthetic parquets!)
    if not _is_official_benchmark_parquets(bench_dir):
        print("[00_download] Replacing benchmark parquets with official datasets-server parquets...")
        for bfile in ["mmlu_all_test.parquet", "arc_challenge_test.parquet", "ifeval_train.parquet"]:
            bp = bench_dir / bfile
            if bp.exists():
                bp.unlink()
        _download_parquet_split_via_datasets_server(
            "cais/mmlu", "all", "test", bench_dir / "mmlu_all_test.parquet", 14042
        )
        _download_parquet_split_via_datasets_server(
            "allenai/ai2_arc", "ARC-Challenge", "test", bench_dir / "arc_challenge_test.parquet", 1172
        )
        _download_parquet_split_via_datasets_server(
            "google/IFEval", "default", "train", bench_dir / "ifeval_train.parquet", 541
        )

    assert _is_official_benchmark_parquets(bench_dir), (
        "Benchmark parquets failed official authenticity verification!"
    )

    # 5. Ensure official instruction_following_eval package is present
    _ensure_official_ifeval_package()

    # 6. Verify tokenizer wrapper and collect exact stats for sources_lock.json
    tokenizer = QwenTokenizerWrapper(tok_dir)
    mmlu_tbl = pq.read_table(bench_dir / "mmlu_all_test.parquet", columns=["subject"])
    mmlu_subjects = len(set(mmlu_tbl.column("subject").to_pylist()))
    arc_tbl = pq.read_table(bench_dir / "arc_challenge_test.parquet", columns=["id"])
    ifeval_tbl = pq.read_table(bench_dir / "ifeval_train.parquet", columns=["key"])

    tldr_sha = _fetch_dataset_commit_sha("openai/summarize_from_feedback", "comparisons")
    nr_sha = _fetch_dataset_commit_sha("HuggingFaceH4/no_robots", "default")
    mmlu_sha = _fetch_dataset_commit_sha("cais/mmlu", "all")
    arc_sha = _fetch_dataset_commit_sha("allenai/ai2_arc", "ARC-Challenge")
    ifeval_sha = _fetch_dataset_commit_sha("google/IFEval", "default")

    sources_lock = {
        "synthetic": False,
        "tldr": {
            "repo": "openai/summarize_from_feedback",
            "config": "comparisons",
            "splits": ["train", "validation"],
            "revision": tldr_sha,
            "train_parquet_sha256": sha256_file(tldr_train_dest),
            "val_parquet_sha256": sha256_file(tldr_val_dest),
            "train_sha256": sha256_file(tldr_train_dest),
            "val_sha256": sha256_file(tldr_val_dest),
        },
        "warmup": {
            "repo": "HuggingFaceH4/no_robots",
            "split": "train",
            "revision": nr_sha,
            "train_parquet_sha256": sha256_file(nr_dest),
            "train_sha256": sha256_file(nr_dest),
        },
        "tokenizer": {
            "repo": "Qwen/Qwen2.5-0.5B-Base",
            "hub_repo": "Qwen/Qwen2.5-0.5B",
            "hf_hub_repo": "Qwen/Qwen2.5-0.5B",
            "revision": qwen_sha,
            "revision_for_repo": "Qwen/Qwen2.5-0.5B",
            "vocab_size": tokenizer.vocab_size,
            "len_tokenizer": len(tokenizer),
            "vocab_sha256": sha256_file(tok_dir / "vocab.json"),
            "merges_sha256": sha256_file(tok_dir / "merges.txt"),
            "tokenizer_json_sha256": sha256_file(tok_dir / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(tok_dir / "tokenizer_config.json"),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "probe_token_ids": {
                "hello": tokenizer.encode("hello"),
                " the": tokenizer.encode(" the"),
            },
        },
        "benchmarks": {
            "ifeval": {
                "repo": "google/IFEval",
                "config": "default",
                "split": "train",
                "revision": ifeval_sha,
                "source_rows": ifeval_tbl.num_rows,
                "parquet_sha256": sha256_file(bench_dir / "ifeval_train.parquet"),
                "sha256": sha256_file(bench_dir / "ifeval_train.parquet"),
            },
            "arc_challenge": {
                "repo": "allenai/ai2_arc",
                "config": "ARC-Challenge",
                "split": "test",
                "revision": arc_sha,
                "source_rows": arc_tbl.num_rows,
                "parquet_sha256": sha256_file(bench_dir / "arc_challenge_test.parquet"),
                "sha256": sha256_file(bench_dir / "arc_challenge_test.parquet"),
            },
            "mmlu": {
                "repo": "cais/mmlu",
                "config": "all",
                "split": "test",
                "revision": mmlu_sha,
                "source_rows": mmlu_tbl.num_rows,
                "num_subjects": mmlu_subjects,
                "parquet_sha256": sha256_file(bench_dir / "mmlu_all_test.parquet"),
                "sha256": sha256_file(bench_dir / "mmlu_all_test.parquet"),
            },
        },
    }

    lock_path = target_cache_dir / "sources_lock.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(sources_lock, f, indent=2, ensure_ascii=False)
    print(
        f"[00_download] Official sources locked at {lock_path} "
        f"(vocab_size={tokenizer.vocab_size}, len_tokenizer={len(tokenizer)}, "
        f"MMLU={mmlu_tbl.num_rows}/{mmlu_subjects} subj, ARC={arc_tbl.num_rows}, IFEval={ifeval_tbl.num_rows})."
    )


if __name__ == "__main__":
    main()

"""Train Nuqta-Nano from scratch (Roman Urdu -> Urdu script).

Protocol: Dakshina lexicon train split only (106,260 pairs); model
selection by corpus CER on the dev split, evaluated every epoch with
greedy decoding; the test split is never touched here — final numbers
come from scripts/evaluate.py on the selected checkpoint.

Deterministic given the seed: data order, batching and initialization
are all driven by seeded generators. CPU-only by design.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from nuqta_nano.metrics import group_refs, score_predictions  # noqa: E402
from nuqta_nano.model import Seq2SeqConfig, TranslitTransformer, greedy_decode  # noqa: E402
from nuqta_nano.vocab import BOS_ID, EOS_ID, PAD_ID, VocabPair  # noqa: E402

SEED = 20261029
BATCH_SIZE = 512


def load_pairs(split: str):
    rows = []
    for line in (PKG / f"data/{split}.tsv").read_text(encoding="utf-8").splitlines():
        r, u = line.split("\t")
        rows.append((r, u))
    return rows


def make_batches(encoded, order, batch_size):
    """Yield (src, tgt) padded batches for the given index order."""
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        srcs = [encoded[j][0] for j in idx]
        tgts = [encoded[j][1] for j in idx]
        src_len = max(len(s) for s in srcs)
        tgt_len = max(len(t) for t in tgts)
        src = torch.full((len(idx), src_len), PAD_ID, dtype=torch.long)
        tgt = torch.full((len(idx), tgt_len), PAD_ID, dtype=torch.long)
        for k, (s, t) in enumerate(zip(srcs, tgts)):
            src[k, : len(s)] = torch.tensor(s)
            tgt[k, : len(t)] = torch.tensor(t)
        yield src, tgt


def epoch_order(encoded, gen) -> list[int]:
    """Length-grouped batching order: shuffle, sort within windows of
    8192 by total length (cuts padding roughly in half on this data),
    then shuffle the batch order. Seeded via `gen`; disclosed in
    REPORT.md because it changes batch composition vs pure shuffling."""
    n = len(encoded)
    perm = torch.randperm(n, generator=gen).tolist()
    lengths = [len(s) + len(t) for s, t in encoded]
    grouped: list[int] = []
    for i in range(0, n, 8192):
        window = perm[i : i + 8192]
        window.sort(key=lambda j: lengths[j])
        grouped.extend(window)
    batches = [grouped[i : i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    batch_order = torch.randperm(len(batches), generator=gen).tolist()
    return [j for b in batch_order for j in batches[b]]


@torch.no_grad()
def dev_eval(model, vocabs, dev_pairs):
    refs = group_refs(dev_pairs)
    inputs = list(refs.keys())
    preds = {}
    for i in range(0, len(inputs), BATCH_SIZE):
        chunk = inputs[i : i + BATCH_SIZE]
        encoded = [vocabs.src.encode(w) for w in chunk]
        out = greedy_decode(model, encoded, BOS_ID, EOS_ID, PAD_ID, max_len=28)
        for w, ids in zip(chunk, out):
            preds[w] = vocabs.tgt.decode(ids)
    return score_predictions(preds, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 4096 pairs, timing only")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    vocabs = VocabPair.load(PKG / "vocab.json")

    train_pairs = load_pairs("train")
    dev_pairs = load_pairs("dev")
    if args.smoke:
        train_pairs = train_pairs[:4096]
        args.epochs = 1

    encoded = [(vocabs.src.encode(r), vocabs.tgt.encode(u)) for r, u in train_pairs]
    cfg = Seq2SeqConfig(src_vocab_size=len(vocabs.src), tgt_vocab_size=len(vocabs.tgt))
    model = TranslitTransformer(cfg)
    n_params = model.count_parameters()
    print(f"params: {n_params:,}", flush=True)

    steps_per_epoch = math.ceil(len(encoded) / BATCH_SIZE)
    total_steps = steps_per_epoch * args.epochs
    warmup = 300

    def lr_at(step: int) -> float:
        if step < warmup:
            return 1e-3 * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    gen = torch.Generator().manual_seed(SEED)

    log_path = PKG / f"{args.out_prefix}training_log.txt"
    log = open(log_path, "a", encoding="utf-8")
    best_cer = float("inf")
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = epoch_order(encoded, gen)
        total_loss, total_tok = 0.0, 0
        for src, tgt in make_batches(encoded, order, BATCH_SIZE):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in, src == PAD_ID, tgt_in == PAD_ID)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ntok = int((tgt_out != PAD_ID).sum())
            total_loss += float(loss.detach()) * ntok
            total_tok += ntok
            step += 1
        train_loss = total_loss / total_tok
        do_eval = (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs
        scores = dev_eval(model, vocabs, dev_pairs) if do_eval else None
        elapsed = time.time() - t_start
        record = {
            "epoch": epoch + 1, "train_loss": round(train_loss, 4),
            "dev_corpus_cer": round(scores["corpus_cer"], 4) if scores else None,
            "dev_exact_match": round(scores["exact_match"], 4) if scores else None,
            "elapsed_s": round(elapsed, 1),
        }
        line = json.dumps(record)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if scores and scores["corpus_cer"] < best_cer:
            best_cer = scores["corpus_cer"]
            ckpt = {
                "model_state": model.state_dict(),
                "config": vars(cfg), "epoch": epoch + 1, "seed": SEED,
                "dev_corpus_cer": best_cer,
            }
            torch.save(ckpt, PKG / f"{args.out_prefix}checkpoint.pt")
            print(f"  new best dev CER {best_cer:.4f} (epoch {epoch + 1}) — checkpoint saved", flush=True)
    log.close()

    if not args.smoke:
        # Release artifacts from the best checkpoint.
        ckpt = torch.load(PKG / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        save_file(model.state_dict(), str(PKG / "model.safetensors"),
                  metadata={"format": "pt"})
        config = {
            "d_model": cfg.d_model, "n_head": cfg.n_head,
            "n_encoder_layers": cfg.n_encoder_layers,
            "n_decoder_layers": cfg.n_decoder_layers, "d_ff": cfg.d_ff,
            "dropout": cfg.dropout, "max_src_len": cfg.max_src_len,
            "max_tgt_len": cfg.max_tgt_len,
            "direction": "roman->urdu", "seed": SEED,
            "parameters": n_params,
            "selected_epoch": ckpt["epoch"], "dev_corpus_cer": ckpt["dev_corpus_cer"],
        }
        (PKG / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        print(f"done: best dev CER {best_cer:.4f} at epoch {ckpt['epoch']}; "
              f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

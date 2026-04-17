"""
RWKV-4-169M 推理引擎

实测 einsum 公式（已验证）：
  # FFN: kx[B,T,H] @ Wfk.t()[H,4H]
  k2 = einsum('bth,hO->btO', kx, Wfk.t())           # → [B,T,4H]
  r[B,T,H] → r_expanded[B,T,4H] via reshape
  gated = r_expanded * k2
  out = einsum('btO,Oh->bth', gated, Wfv.t())       # → [B,T,H]
  # Attention: (r*wkv)[B,T,H] @ Wo.t()[H,H]
  att_out = einsum('bth,hO->btO', r*wkv, Wo.t())   # → [B,T,H]
"""

import os, sys, time, json, torch, torch.nn.functional as F
sys.path.insert(0, "D:/pylib")
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from typing import Optional, Tuple, List

MODEL_PATH     = os.environ.get("RWKV_MODEL",     "D:/pylib/rwkv-4-169m-native.pth")
TOKENIZER_PATH = os.environ.get("RWKV_TOKENIZER", "D:/pylib/tokenizer.json")


class RWKVTokenizer:
    """rwkv 官方 TRIE_TOKENIZER"""
    def __init__(self):
        vocab = "D:/pylib/rwkv4-world-tok/rwkv_vocab_v20230424.txt"
        self._t = TRIE_TOKENIZER(vocab)

    def encode(self, text: str) -> List[int]:
        return self._t.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._t.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._t.idx2token)


class RWKV4:
    def __init__(self, state_dict: dict):
        self.sd = state_dict
        self.n_layers = sum(1 for k in state_dict if ".ln1.weight" in k)
        self.hidden = state_dict["emb.weight"].shape[1]   # 768
        self.vocab  = state_dict["emb.weight"].shape[0]  # 50277
        n_params = sum(v.numel() for v in state_dict.values())
        print(f"RWKV-4-{n_params/1e6:.0f}M | layers={self.n_layers} "
              f"hidden={self.hidden} vocab={self.vocab}")

    def _lm(self, x, w, b):
        """LayerNorm: x[B,T,H], w/b[H]"""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * w * (var + 1e-5).rsqrt() + b

    def _ln(self, x, w, b):
        return self._lm(x, w, b)

    def _get_state(self, B, device):
        H, nL = self.hidden, self.n_layers
        return {
            "xx": torch.zeros(B, nL, H, device=device),
            "aa": torch.zeros(B, nL, H, device=device),
            "bb": torch.zeros(B, nL, H, device=device),
            "pp": torch.full((B, nL, H), -1e30, device=device),
        }

    def forward(self, tokens: torch.Tensor,
                state: Optional[dict] = None
                ) -> Tuple[torch.Tensor, dict]:
        sd = self.sd
        H, V, nL = self.hidden, self.vocab, self.n_layers
        B, T = tokens.shape
        dev = tokens.device

        if state is None:
            state = self._get_state(B, dev)

        # Clip token IDs to valid vocab range (tokenizer may produce IDs > vocab_size)
        x = F.embedding(tokens, sd["emb.weight"], padding_idx=-1)

        for i in range(nL):
            p = f"blocks.{i}."

            # Pre-LN (ln0)
            ln0_w = sd.get(p + "ln0.weight")
            if ln0_w is not None:
                x = self._ln(x, ln0_w, sd[p + "ln0.bias"])

            xx_prev = state["xx"][:, i, :]
            aa_prev = state["aa"][:, i, :]
            bb_prev = state["bb"][:, i, :]
            pp_prev = state["pp"][:, i, :]

            # ---- Attention ----
            tm_k = sd[p + "att.time_mix_k"].squeeze()
            tm_v = sd[p + "att.time_mix_v"].squeeze()
            tm_r = sd[p + "att.time_mix_r"].squeeze()
            decay   = sd[p + "att.time_decay"].float()
            t_first = sd[p + "att.time_first"].float()

            kx = x * tm_k + xx_prev.view(B, 1, H) * (1 - tm_k)
            vx = x * tm_v + xx_prev.view(B, 1, H) * (1 - tm_v)
            rx = x * tm_r + xx_prev.view(B, 1, H) * (1 - tm_r)

            Wr = sd[p + "att.receptance.weight"]
            Wk = sd[p + "att.key.weight"]
            Wv = sd[p + "att.value.weight"]
            Wo = sd[p + "att.output.weight"]

            # r = sigmoid(rx @ Wr)
            r = torch.sigmoid(torch.einsum('bth,hh->bth', rx, Wr))
            k = torch.einsum('bth,hh->bth', kx, Wk).float()
            v = torch.einsum('bth,hh->bth', vx, Wv).float()

            # WKV accumulation
            aa_cur = aa_prev.clone()
            bb_cur = bb_prev.clone()
            pp_cur = pp_prev.clone()

            for t in range(T):
                kk = k[:, t, :]
                vv = v[:, t, :]
                rr = r[:, t, :]
                ww = t_first + kk
                p_max = torch.maximum(pp_cur, ww)
                e1 = torch.exp(pp_cur - p_max)
                e2 = torch.exp(ww - p_max)
                wkv = ((e1 * aa_cur + e2 * vv) / (e1 * bb_cur + e2 + 1e-8)).to(x.dtype)
                aa_cur = e1 * aa_cur + e2 * vv
                bb_cur = e1 * bb_cur + e2
                pp_cur = torch.maximum(decay + pp_cur, kk)

            # Attention output: einsum('bth,hh->bth', r*wkv, Wo)
            wkv_out = torch.einsum('bth,hh->bth', r[:, -1:, :] * wkv.unsqueeze(1), Wo)
            att_out = wkv_out.to(x.dtype)

            # ---- FFN ----
            ffn_ln = self._ln(x, sd[p + "ln2.weight"], sd[p + "ln2.bias"])
            fk = sd[p + "ffn.time_mix_k"].squeeze()
            fr = sd[p + "ffn.time_mix_r"].squeeze()
            Wfk = sd[p + "ffn.key.weight"]          # [4H, H]
            Wfv = sd[p + "ffn.value.weight"]        # [H, 4H]
            Wfr = sd[p + "ffn.receptance.weight"]  # [H, H]

            ffn_kx = ffn_ln * fk + xx_prev.view(B, 1, H) * (1 - fk)
            ffn_rx = ffn_ln * fr + xx_prev.view(B, 1, H) * (1 - fr)

            r_ffn = torch.sigmoid(torch.einsum('bth,hh->bth', ffn_rx, Wfr))
            # einsum: kx[B,T,H] @ Wfk.t()[H,4H] → k2[B,T,4H]
            k2 = torch.einsum('bth,hO->btO', ffn_kx, Wfk.t())
            k2 = F.relu(k2).square()
            # r_ffn [B,T,H] → expand to [B,T,4H]
            r4 = r_ffn.unsqueeze(-1).expand(B, T, H, 4).reshape(B, T, 4 * H)
            gated = r4 * k2
            # einsum: gated[B,T,4H] @ Wfv.t()[4H,H] → ffn_out[B,T,H]
            ffn_out = torch.einsum('btO,Oh->bth', gated, Wfv.t())
            ffn_out = ffn_out.to(x.dtype)

            # ln1 + residual
            x = x + att_out + ffn_out
            x = self._ln(x, sd[p + "ln1.weight"], sd[p + "ln1.bias"])
            state["xx"][:, i, :] = x[:, -1, :]

        # Final LN + head
        ln_out_w = sd.get("ln_out.weight", torch.ones(H, device=dev))
        ln_out_b = sd.get("ln_out.bias",   torch.zeros(H, device=dev))
        x = self._ln(x, ln_out_w, ln_out_b)
        logits = torch.einsum('bth,vh->btv', x, sd["head.weight"])
        return logits, state

    def __call__(self, tokens, state=None):
        return self.forward(tokens, state)


def load_model(path: str = MODEL_PATH) -> Tuple[RWKV4, dict]:
    print(f"Loading: {path}")
    t0 = time.time()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} keys | {time.time()-t0:.1f}s")
    return RWKV4(sd), sd


def generate(model: RWKV4, tokenizer: RWKVTokenizer,
            prompt: str, max_new: int = 50,
            temperature: float = 0.7, top_p: float = 0.9) -> str:
    tokens = tokenizer.encode(prompt)
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = None
    with torch.no_grad():
        logits, state = model(tok_tensor, state)
    output = list(tokens)
    for _ in range(max_new):
        logits_last = logits[0, -1, :].float() / max(temperature, 0.01)
        sorted_lp, sorted_idx = torch.sort(logits_last, descending=True)
        probs = torch.softmax(sorted_lp, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > top_p
        mask[0] = False
        sorted_lp = sorted_lp.masked_fill(mask, float("-inf"))
        probs = torch.softmax(sorted_lp, dim=-1)
        try:
            next_tok = sorted_idx[torch.multinomial(probs, 1).item()].item()
        except Exception:
            next_tok = sorted_idx[0].item()
        output.append(next_tok)
        tok_tensor = torch.tensor([[next_tok]], dtype=torch.long)
        with torch.no_grad():
            logits, state = model(tok_tensor, state)
        if next_tok == 0:
            break
    return tokenizer.decode(output[len(tokens):])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="def hello():")
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()
    model, _ = load_model()
    tok = RWKVTokenizer()
    t0 = time.time()
    out = generate(model, tok, args.prompt, args.max_tokens, args.temperature)
    # ASCII-only safe output
    safe_out = "".join(c if 32 <= ord(c) < 127 else "." for c in out)
    print(f"\n[{args.prompt}]")
    print(f"  -> ASCII: {safe_out!r}")
    print(f"  ({time.time()-t0:.2f}s, {args.max_tokens} tokens)")

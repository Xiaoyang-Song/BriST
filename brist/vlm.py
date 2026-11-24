"""SmolVLM wrapper: 4-bit (QLoRA) decoder, trainable projector, token-level I/O.

Both pathways feed the *same* chat prompt; the 64 image-token embeddings are replaced
either by CT tokens Z_U = g_U(f_U(U)) or by signal tokens Z_X = g_X(f_X(X)).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
STYLES = {"A": "detailed", "B": "concise", "C": "structured"}
LORA_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION_TARGETS = r".*vision_model.*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)"


class VLM(nn.Module):
    def __init__(self, device="cuda", lora_r=16, lora_alpha=32, train_connector=True, vision_lora=False):
        super().__init__()
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.tok = self.proc.tokenizer
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                                 llm_int8_skip_modules=["vision_model", "connector", "lm_head"])
        m = Idefics3ForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                             torch_dtype=torch.float16, device_map={"": 0})
        targets = f"({LORA_TARGETS})|({VISION_TARGETS})" if vision_lora else LORA_TARGETS
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=targets)
        self.pm = get_peft_model(m, cfg)
        self.m = m
        self.connector = m.model.connector
        if train_connector:
            self.connector.float()
            for p in self.connector.parameters():
                p.requires_grad = True
        self.text = m.model.text_model
        self.lm_head = m.lm_head
        self.d = m.config.text_config.hidden_size
        self.image_token_id = m.config.image_token_id
        self.eou = self.tok.convert_tokens_to_ids("<end_of_utterance>")
        self.prompts, self.img_pos = {}, {}
        dummy = Image.new("RGB", (512, 512))
        for s, name in STYLES.items():
            msgs = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Describe the build quality of this region. Write a {name} report."}]}]
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
            ids = self.proc(text=text, images=[dummy], images_kwargs=dict(do_image_splitting=False),
                            return_tensors="pt")["input_ids"][0]
            self.prompts[s] = ids.to(device)
            self.img_pos[s] = (ids == self.image_token_id).nonzero().squeeze(1).to(device)
        assert len(self.img_pos["A"]) == 64

    # ------------------------------------------------------------ vision pathway
    @torch.no_grad()
    def vision_features(self, images):
        pv = self.proc.image_processor(images, do_image_splitting=False, return_tensors="pt")["pixel_values"]
        pv = pv[:, 0].to(self.device, torch.float16)
        return self.m.model.vision_model(pixel_values=pv).last_hidden_state   # (B,1024,768)

    def encode_pixels(self, img_uint8):
        """img_uint8: (B,512,512,3) uint8 tensor/array of CT renderings -> vision tokens H_U (with grad
        through the vision LoRA). SigLIP preprocessing: rescale to [0,1], normalise with mean=std=0.5."""
        x = torch.as_tensor(img_uint8, device=self.device).permute(0, 3, 1, 2).to(torch.float16)
        pv = (x / 255.0 - 0.5) / 0.5
        return self.m.model.vision_model(pixel_values=pv).last_hidden_state

    def vision_lora_params(self):
        return [p for n, p in self.pm.named_parameters() if p.requires_grad and "lora_" in n and "vision_model" in n]

    def connect(self, H):
        return self.connector(H.to(next(self.connector.parameters()).dtype))  # (B,64,d)

    # ------------------------------------------------------------ text helpers
    def encode_answers(self, texts):
        return [self.tok(" " + t, add_special_tokens=False)["input_ids"] + [self.eou] for t in texts]

    def _prefix(self, Z, styles):
        emb = self.text.get_input_embeddings()
        out = []
        for z, s in zip(Z, styles):
            e = emb(self.prompts[s]).clone()
            e[self.img_pos[s]] = z.to(e.dtype)
            out.append(e)
        return out

    def forward_logits(self, Z, styles, answer_ids):
        """Teacher-forced pass. Returns logits over answer positions (B, Lmax, V) and mask (B, Lmax)."""
        emb = self.text.get_input_embeddings()
        pre = self._prefix(Z, styles)
        P = [p.shape[0] for p in pre]
        L = [len(a) for a in answer_ids]
        T = max(p + l for p, l in zip(P, L))
        B = len(pre)
        x = torch.zeros(B, T, self.d, device=self.device, dtype=pre[0].dtype)
        att = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, (p, a) in enumerate(zip(pre, answer_ids)):
            a = torch.tensor(a, device=self.device)
            x[i, :len(p)] = p
            x[i, len(p):len(p) + len(a)] = emb(a)
            att[i, :len(p) + len(a)] = 1
        h = self.text(inputs_embeds=x, attention_mask=att).last_hidden_state
        Lm = max(L)
        # hidden state at position t predicts token t+1: answer token j is predicted from P+j-1
        idx = torch.stack([torch.arange(Lm, device=self.device) + p - 1 for p in P])
        hs = torch.gather(h, 1, idx.clamp(max=T - 1)[..., None].expand(-1, -1, h.shape[-1]))
        logits = self.lm_head(hs).float()
        mask = torch.zeros(B, Lm, dtype=torch.bool, device=self.device)
        tgt = torch.full((B, Lm), -100, dtype=torch.long, device=self.device)
        for i, a in enumerate(answer_ids):
            mask[i, :len(a)] = True
            tgt[i, :len(a)] = torch.tensor(a, device=self.device)
        return logits, mask, tgt

    def lm_loss(self, Z, styles, texts=None, answer_ids=None):
        if answer_ids is None:
            answer_ids = self.encode_answers(texts)
        logits, mask, tgt = self.forward_logits(Z, styles, answer_ids)
        return F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100)

    @torch.no_grad()
    def generate(self, Z, style="A", max_new=110):
        """Greedy decoding with KV cache; all prompts share length so no padding is needed."""
        B = Z.shape[0]
        x = torch.stack(self._prefix(Z, [style] * B))
        out = self.text(inputs_embeds=x, use_cache=True)
        past = out.past_key_values
        nxt = self.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
        toks = [nxt]
        done = nxt == self.eou
        emb = self.text.get_input_embeddings()
        for _ in range(max_new - 1):
            out = self.text(inputs_embeds=emb(nxt)[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = self.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eou), nxt)
            toks.append(nxt)
            done = done | (nxt == self.eou)
            if done.all():
                break
        toks = torch.stack(toks, 1).tolist()
        res = []
        for t in toks:
            if self.eou in t:
                t = t[:t.index(self.eou)]
            res.append(self.tok.decode(t, skip_special_tokens=True).strip())
        return res

    # ------------------------------------------------------------ params
    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.pm.named_parameters() if p.requires_grad}

    def load_trainable(self, sd, strict_lora=True):
        own = dict(self.pm.named_parameters())
        n_loaded = 0
        for n, v in sd.items():
            if n in own:
                own[n].data.copy_(v.to(own[n].dtype))
                n_loaded += 1
        assert n_loaded > 0
        return n_loaded

    def lora_params(self):
        return [p for n, p in self.pm.named_parameters() if p.requires_grad and "lora_" in n
                and "vision_model" not in n]

    def connector_params(self):
        return [p for p in self.connector.parameters() if p.requires_grad]

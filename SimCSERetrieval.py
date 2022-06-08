# -*- coding: utf-8 -*-
# @Time    : 2022/06/08
# @Author  : xiximayou

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSE import SimCSE
from torch import Tensor


def cos_sim(a: Tensor, b: Tensor, device):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1).to(device)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1).to(device)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class SimCSERetrieval(object):
    def __init__(self,
                 fname,
                 pretrained_path,
                 simcse_model_path,
                 batch_size=32,
                 max_length=100,
                 device="cuda"):
        self.fname = fname
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        model = SimCSE(pretrained=pretrained_path).to(device)
        model.load_state_dict(torch.load(simcse_model_path, map_location=device))
        self.model = model
        self.model.eval()
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode(self, text):
        text_encs = self.tokenizer(text,
                                   padding=True,
                                   max_length=self.max_length,
                                   truncation=True,
                                   return_tensors="pt")
        input_ids = text_encs["input_ids"].to(self.device)
        attention_mask = text_encs["attention_mask"].to(self.device)
        token_type_ids = text_encs["token_type_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.forward(input_ids, attention_mask, token_type_ids)
        return output

    def sim(self, text1, text2):
        text1_vec = self.encode(text1)
        text2_vec = self.encode(text2)
        score = cos_sim(text1_vec, text2_vec, self.device)
        return score.item()

    def encode_batch(self, texts):
        text_encs = self.tokenizer(texts,
                                   padding=True,
                                   max_length=self.max_length,
                                   truncation=True,
                                   return_tensors="pt")
        input_ids = text_encs["input_ids"].to(self.device)
        attention_mask = text_encs["attention_mask"].to(self.device)
        token_type_ids = text_encs["token_type_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.forward(input_ids, attention_mask, token_type_ids)
        return output

    def encode_file(self, save=False, save_file=None):
        all_texts = []
        all_ids = []
        all_vecs = []
        import pickle
        if not os.path.exists(save_file):
            with open(self.fname, "r", encoding="utf8") as h:
                texts = []
                idxs = []
                h = h.readlines()
                for idx, line in enumerate(tqdm(h, ncols=100)):
                    if not line.strip():
                        continue
                    texts.append(line.strip())
                    idxs.append(idx)
                    if len(texts) >= self.batch_size:
                        vecs = self.encode_batch(texts)
                        vecs = vecs / vecs.norm(dim=1, keepdim=True)
                        all_texts.extend(texts)
                        all_ids.extend(idxs)
                        all_vecs.append(vecs.cpu())
                        texts = []
                        idxs = []
                if texts:
                    vecs = self.encode_batch(texts)
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                    all_texts.extend(texts)
                    all_ids.extend(idxs)
                    all_vecs.append(vecs.cpu())
            all_vecs = torch.cat(all_vecs, 0).numpy()
            id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
            self.id2text = id2text
            self.vecs = all_vecs
            self.ids = np.array(all_ids, dtype="int64")
            if save:
                with open(save_file, 'wb') as fp:
                    pickle.dump([self.id2text, self.vecs, self.ids], fp)
        else:
            print("加载保存好的向量")
            with open(save_file, 'rb') as fp:
                self.id2text, self.vecs, self.ids = pickle.load(fp)

    def build_index(self, nlist=256):
        import faiss
        dim = self.vecs.shape[1]
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(nlist, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def sim_query_ori(self, sentence, topK=10):
        query_vec = self.encode_batch(sentence)
        maxk = max((topK,))
        top = cos_sim(query_vec, self.vecs, self.device).topk(maxk, dim=1, largest=True, sorted=True)
        values, indices = top
        # print('query：', sentence)
        # print("doc：")
        sents = []
        for val, ind in zip(values[0], indices[0]):
            ind = ind.item()
            sents.append(self.id2text[ind])
        return sents

    def sim_query(self, sentence, topK=20):
        vec = self.encode_batch([sentence])
        vec = vec / vec.norm(dim=1, keepdim=True)
        vec = vec.cpu().numpy()
        _, sim_idx = self.index.search(vec, topK)
        sim_sentences = []
        for i in range(sim_idx.shape[1]):
            idx = sim_idx[0, i]
            sim_sentences.append(self.id2text[idx])
        return sim_sentences

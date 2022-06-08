# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSE import SimCSE
from CSECollator import CSECollator
from tensorboardX import SummaryWriter

tensorboard_dir = 'output/tensorboard/'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
train_writer = SummaryWriter(log_dir=tensorboard_dir)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, help="train text file")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./model", help="model output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    args = parser.parse_args()
    return args


# 这里使用了datasets库进行数据的预处理
def load_data(args, tokenizer):
    data_files = {"train": args.train_file}
    ds = load_dataset("text", data_files=data_files)
    ds_tokenized = ds.map(lambda example: tokenizer(example["text"]), num_proc=args.num_proc)
    collator = CSECollator(tokenizer, max_len=args.max_length)
    dl = DataLoader(ds_tokenized["train"],
                    batch_size=args.batch_size,
                    collate_fn=collator.collate)
    return dl


# def compute_loss(y_pred, tao=0.05, device="cuda"):
#     idxs = torch.arange(0, y_pred.shape[0], device=device)
#     y_true = idxs + 1 - idxs % 2 * 2
#     similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
#     similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
#     similarities = similarities / tao
#     loss = F.cross_entropy(similarities, y_true)
#     return torch.mean(loss)


def compute_loss(y_pred, lamda=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)  # [0,1,2,3,4,5] 
    # 这里[(0,1),(2,3),(4,5)]代表三组样本，
    # 其中0,1是同一个句子，输入模型两次
    # 其中2,3是同一个句子，输入模型两次
    # 其中4,5是同一个句子，输入模型两次
    y_true = idxs + 1 - idxs % 2 * 2  # 生成真实的label  = [1,0,3,2,5,4]  
    # 计算各句子之间的相似度，形成下方similarities 矩阵，其中xij 表示第i句子和第j个句子的相似度
    # [[ x00,x01,x02,x03,x04 ,x05  ]
    # [ x10,x11,x12,x13,x14 ,x15  ]
    # [ x20,x21,x22,x23,x24 ,x25  ]
    # [ x30,x31,x32,x33,x34 ,x35  ]
    # [ x40,x41,x42,x43,x44 ,x45  ]
    # [ x50,x51,x52,x53,x54 ,x55  ]]
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # similarities屏蔽对角矩阵即自身相等的loss
    # [[ -nan,x01,x02,x03,x04 ,x05   ]   
    # [ x10, -nan,x12,x13,x14 ,x15 ]
    # [ x20,x21, -nan,x23,x24 ,x25 ]
    # [ x30,x31,x32, -nan,x34 ,x35 ]
    # [ x40,x41,x42,x43, -nan,x45  ]
    # [ x50,x51,x52,x53,x54 , -nan ]]
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 论文中除以 temperature 超参
    similarities = similarities / lamda
    # 下面这一行计算的是相似矩阵每一行和y_true = [1,0,3,2,5,4] 的交叉熵损失
    # [[ -nan,x01,x02,x03,x04 ,x05  ]   label = 1 含义：第0个句子应该和第1个句子的相似度最高，即x01越接近1越好
    # [ x10, -nan,x12,x13,x14,x15 ]   label = 0  含义：第1个句子应该和第0个句子的相似度最高，即x10越接近1越好
    # [ x20,x21, -nan,x23,x24,x25 ]   label = 3  含义：第2个句子应该和第3个句子的相似度最高，即x23越接近1越好
    # [ x30,x31,x32, -nan,x34,x35 ]   label = 2  含义：第3个句子应该和第2个句子的相似度最高，即x32越接近1越好
    # [ x40,x41,x42,x43, -nan,x45 ]   label = 5  含义：第4个句子应该和第5个句子的相似度最高，即x45越接近1越好
    # [ x50,x51,x52,x53,x54 , -nan ]]  label = 4 含义：第5个句子应该和第4个句子的相似度最高，即x54越接近1越好
    # 这行代码就是simsce的核心部分，就是一个句子被dropout 两次得到的向量相似度应该越大 
    # 越好，且和其他句子向量的相似度越小越好
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained, mirror="tuna")
    dl = load_data(args, tokenizer)
    model = SimCSE(args.pretrained, args.pool_type, args.dropout_rate).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_out = Path(args.model_out)
    if not model_out.exists():
        os.mkdir(model_out)

    model.train()
    batch_idx = 0
    for epoch_idx in range(args.epochs):
        for data in tqdm(dl):
            batch_idx += 1
            pred = model(input_ids=data["input_ids"].to(args.device),
                         attention_mask=data["attention_mask"].to(args.device),
                         token_type_ids=data["token_type_ids"].to(args.device))
            loss = compute_loss(pred, args.tao, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            if batch_idx % args.display_interval == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")
                train_writer.add_scalar("loss", loss, batch_idx)
            # if batch_idx % args.save_interval == 0:
                # torch.save(model.state_dict(), model_out / "epoch_{0}-batch_{1}-loss_{2:.6f}".format(epoch_idx, batch_idx, loss))
    torch.save(model.state_dict(), model_out / "epoch_{}-loss_{:.6f}".format(epoch_idx, loss))


def main():
    args = parse_args()
    logging.info(vars(args))
    train(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

# -*- coding: utf-8 -*-
# @Time    : 2022/06/08
# @Author  : xiximayou

import logging
import torch
from SimCSERetrieval import SimCSERetrieval


def main():
    fname = "./data/news_title.txt"
    pretrained = "./model_hub/chinese-bert-wwm-ext"  # huggingface modelhub 下载的预训练模型
    simcse_model = "./model/epoch_1-loss_0.000622"
    batch_size = 64
    max_length = 100
    use_fasii = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device)

    logging.info("Sentences to vectors")
    simcse.encode_file(save=True, save_file='output/vectors.pkl')

    if use_fasii:
        logging.info("Build faiss index")
        simcse.build_index(n_list=1024)
        simcse.index.nprob = 20

        query_sentence = "基金亏损路未尽 后市看法仍偏谨慎"
        print("\nquery title:{0}".format(query_sentence))
        print("\nsimilar titles:")
        print(u"\n".join(simcse.sim_query(query_sentence, topK=10)))

    else:
        query_sentence = "基金亏损路未尽 后市看法仍偏谨慎"
        print('得到的句向量：')
        print(simcse.encode(query_sentence).shape)

        print("")
        query_sentence2 = "海通证券：私募对后市看法偏谨慎"
        score = simcse.sim(query_sentence, query_sentence2)
        print("text1：", query_sentence)
        print("text2：", query_sentence2)
        print("相似度：", score)

        print("")
        print("\nquery title:{0}".format(query_sentence))
        print("\nsimilar titles:")
        print(u"\n".join(simcse.sim_query_ori(query_sentence, topK=10)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

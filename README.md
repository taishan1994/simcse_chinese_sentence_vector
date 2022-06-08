# simcse_chinese_sentence_vector
基于simcse的中文句向量生c成。

# 说明

本项目是基于[KwangKa/SIMCSE_unsup: 中文无监督SimCSE Pytorch实现 (github.com)](https://github.com/KwangKa/SIMCSE_unsup)进行的修改。主要改动的地方如下：

- 在SimCSERetrieval.py里面对文件里的句子进行向量化会丢弃最后部分句子，这里修改后得以保全。同时，不使用faiss库进行相似句子识别，新增基于torch的方法。
- 在SimCSERetrieval.py里面新增输入一个句子，获取其句向量的方法，即encode()。新增计算两个句子之间的相似度，即sim()。新增从句向量库中找出和一个句子相似的的句子，即sim_query_ori()。
- 具体的使用用例在test_unsup.py里面。可以设置参数将生成的句向量库通过二进制的格式保存，避免每次重复生成。

数据及相关的模型下载地址：<br>

链接：https://pan.baidu.com/s/1lj5rMBVbG0uWog3FNQQ_Pw?pwd=vomc <br>
提取码：vomc<br>

# 依赖

```
transformers==4.4.0
pytorch==1.6.0
datasets
transorboardX
```



# 训练句向量

数据为data/news_title.txt，一行就是一个句子。

```python
python train_unsup.py --train_file data/news_title.txt --pretrained model_hub/chinese-bert-wwm-ext/
```

# 句向量的使用

这里贴下基本代码：

```python
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

```

#### 运行指令

```python
python test_unsup.py
```

#### 结果

```
2022-06-08 07:35:02,062|root|INFO|Load model
2022-06-08 07:35:09,188|root|INFO|Sentences to vectors
加载保存好的向量
得到的句向量：
torch.Size([1, 768])

text1： 基金亏损路未尽 后市看法仍偏谨慎
text2： 海通证券：私募对后市看法偏谨慎
相似度： 0.7005364298820496


query title:基金亏损路未尽 后市看法仍偏谨慎

similar titles:
基金亏损路未尽 后市看法仍偏谨慎
海通证券：私募对后市看法偏谨慎
基金谨慎看待后市行情
连塑基本面不容乐观 后市仍有下行空间
稳健投资者继续保持观望 市场走势还未明朗
下半年基金投资谨慎乐观
华安基金许之彦：下半年谨慎乐观
前期乐观预期被否 基金重归谨慎
基金公司看后市：南方大成乐观 博时华安谨慎
基金公司谨慎看多明年市场
```

# 补充

Q：怎么训练得到自己文本的句向量？<br>

A：参考news_title.txt里面，每一行一句话。<br>

Q：我的query是句子，语义库是由文档构成的，怎么获取文档的向量？<br>

A：或许可以对文档进行分句，然后得到每一句的句向量，最后对这些句向量进行融合。<br>

Q：获取的向量库应用？<br>

A：可以制作成基于检索的问答系统，可参考[taishan1994/WebQA_tfidf: 针对于百度WebQA数据集，利用TF-IDF等模型构建的问答系统 (github.com)](https://github.com/taishan1994/WebQA_tfidf)。这里使用的是基于tf-idf的检索，或许可以替换成这里的句向量的检索。

# 参考

> https://github.com/KwangKa/SIMCSE_unsup


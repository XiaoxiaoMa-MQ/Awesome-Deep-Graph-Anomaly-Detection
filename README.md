# Awesome-Deep-Graph-Anomaly-Detection

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection) ![GitHub stars](https://img.shields.io/github/stars/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection?color=yellow&label=Stars) ![GitHub forks](https://img.shields.io/github/forks/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection?color=blue&label=Forks) 

A collection of papers on deep learning for graph anomaly detection, and published algorithms and datasets.

- [Awesome-Deep-Graph-Anomaly-Detection](#awesome-deep-graph-anomaly-detection)
  - [A Timeline of graph anomaly detection](#a-timeline-of-graph-anomaly-detection)
  - [Survey](#survey)
  - [Anomalous Node Detection on Static Graphs](#anomalous-node-detection-on-static-graphs)
    - [Anomalous Node Detection on Static Plain Graphs](#anomalous-node-detection-on-static-plain-graphs)
      - [Traditional Non-Deep Learning Techniques](#traditional-non-deep-learning-techniques)
      - [Network Representation Based Techniques](#network-representation-based-techniques)
    - [Anomalous Node Detection on Static Attributed Graphs](#anomalous-node-detection-on-static-attributed-graphs)
      - [Traditional Non-Deep Learning Techniques](#traditional-non-deep-learning-techniques-1)
      - [Deep Neural Network Based Techniques](#deep-neural-network-based-techniques)
      - [Graph Convolutional Neural Network Based Techniques](#graph-convolutional-neural-network-based-techniques)
      - [Graph Attention Neural Network Based Techniques](#graph-attention-neural-network-based-techniques)
      - [Generative Adversarial Neural Network Based Techniques](#generative-adversarial-neural-network-based-techniques)
      - [Reinforcement Learning Based Techniques](#reinforcement-learning-based-techniques)
      - [Network Representation Based Techniques](#network-representation-based-techniques-1)
    - [Anomalous Node Detection on Dynamic Graphs](#anomalous-node-detection-on-dynamic-graphs)
  - [Anomalous Edge Detection](#anomalous-edge-detection)
  - [Anomalous Sub-graph Detection](#anomalous-sub-graph-detection)
  - [Anomalous Graph Detection](#anomalous-graph-detection)
  - [Published Algorithms and Models](#published-algorithms-and-models)
  - [Datasets](#datasets)
    - [Citation/Co-authorship Networks](#citationco-authorship-networks)
    - [Social Networks](#social-networks)
    - [Co-purchasing Networks](#co-purchasing-networks)
    - [Transportation Networks](#transportation-networks)
  - [Tools](#tools)

----------
## A Timeline of graph anomaly detection
[![timeline](https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection/raw/main/Timeline.png)](https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection)

## Survey
__A Comprehensive Survey on Graph Anomaly Detection with Deep Learning__. **28 Pages**, IEEE Trans. Knowl. Data Eng., 2021. 
_Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z. Sheng, Hui Xiong, Leman Akoglu_, [[Paper](https://www.computer.org/csdl/journal/tk/5555/01/09565320/1xx849OoPks)] [[arXiv](https://arxiv.org/abs/2106.07178)]

Link: https://www.computer.org/csdl/journal/tk/5555/01/09565320/1xx849OoPks

    @article{ma2021comprehensive,
        title={A comprehensive survey on graph anomaly detection with 
                deep learning},
        author={Ma, Xiaoxiao and Wu, Jia and Xue, Shan and Yang, Jian and 
                Zhou, Chuan and Sheng, Quan Z and Xiong, Hui and
                Akoglu, Leman},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2021},
        publisher={IEEE}
    }

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Anomaly detection: A survey | ACM Comput. Surv. | 2009 | _Chandola et al._ | [[Paper](https://www.sciencedirect.com/science/article/pii/S1574013720303865)] |
| Outlier detection: Methods, models, and classification | ACM Comput. Surv. | 2020 |  _Boukerche et al._ | [[Paper](https://www.sciencedirect.com/science/article/pii/S1084804518300560)] |
| Anomalous Example Detection in Deep Learning: A Survey | arXiv | 2021 | _Bulusu et al._ | [[Paper](https://arxiv.org/abs/2003.06979)] |
| A comprehensive survey of anomaly detection techniques for high dimensional big data | J. Big Data | 2020 | _Thudumu et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Deep learning for anomaly detection | ACM Comput. Surv. | 2021 | _Pang et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Deep learning for anomaly detection: A survey | arXiv | 2019 | _Chalapathy and Chawla_ | [[Paper](https://arxiv.org/abs/1901.03407)] |
| Graph based anomaly detection and description: A survey | Data Min. Knowl. Discovery | 2015 | _Akoglu et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Anomaly detection in dynamic networks: A survey | Wiley Interdiscip. Rev. Comput. Stat. | 2018 | _Ranshous et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Anomaly detection for big data using efficient techniques: A review | AIDE | 2021 | _Jennifer and Kumar_ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Machine learning techniques for network anomaly detection: A survey | Int. Conf. Inform. IoT Enabling Technol | 2020 | _Eltanbouly et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| A comprehensive survey on network anomaly detection | Telecommun. Syst. | 2020 | _Fernandes et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| A survey of deep learning-based network anomaly detection | Clust. Comput. | 2019 | _Kwon et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| A survey of outlier detection methods in network anomaly identification | Comput. J. | 2011 | _Gogoi et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Anomaly detection in online social networks | Soc. Networks | 2014 | _Savage et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| A survey on social media anomaly detection | SIGKDD Explor. | 2016 | _Yu et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Combining machine learning with knowledge engineering to detect fake news in social networks-a survey, | AAAI Conf. Artif. Intell | 2019 | _Hunkelmann et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Fraud detec- tion: A systematic literature review of graph-based anomaly detection approaches | Decis. Support Syst. | 2020 | _Pourhabibi et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |

----------

## Anomalous Node Detection on Static Graphs

### Anomalous Node Detection on Static Plain Graphs

#### Traditional Non-Deep Learning Techniques

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Oddball: Spotting anomalies in weighted graphs | Pacific-Asia Conf. Knowl. Discov. Data Mining. | 2016 | _Akoglu et al._ | [[Paper]()] |
| Fraudar: Bounding graph fraud in the face of camouflage | ACM SIGKDD | 2016 | _Hooi et al._ | [[Paper]()] |
| Intrusion as (anti)social communication: characterization and detection | ACM SIGKDD | 2012 | _Ding et al._ | [[Paper]()] |

----------

#### Network Representation Based Techniques
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| An embedding approach to anomaly detection | Int. Conf. Data Eng. | 2016 | _Hu et al._ | [[Paper]()] |
| Decoupling representation learning and classification for gnn-based anomaly detection | Int. ACM SIGIR | 2021 | _Wang et al._ | [[Paper]()] |

----------

### Anomalous Node Detection on Static Attributed Graphs

#### Traditional Non-Deep Learning Techniques
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Accelerated local anomaly detection via resolving attributed networks | Int. Joint Conf. Artif. Intell. | 2017 | _Liu et al._ | [[Paper]()] |
| Radar: Residual analysis for anomaly detection in attributed networks | Int. Joint Conf. Artif. Intell., | 2017 | _Li et al._ | [[Paper]()] |
| Anomalous: A joint modeling approach for anomaly detection on attributed networks | Int. Joint Conf. Artif. Intell. | 2018 | _Peng et al._ | [[Paper]()] |

----------

#### Deep Neural Network Based Techniques
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Outlier resistant unsupervised deep architectures for attributed network embedding | Int. Conf. Web Search Data Mining | 2020 | _Bandyopadhyay et al._ | [[Paper]()] |

----------

#### Graph Convolutional Neural Network Based Techniques

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Deep anomaly detection on attributed networks | SIAM Int. Conf. Data Mining | 2019 | _Ding et al._ | [[Paper]()] |
| A deep multi-view framework for anomaly detection on attributed networks | IEEE Trans. Knowl. Data Eng. | 2020 | _Peng et al._ | [[Paper]()] |
| Specae: Spectral autoencoder for anomaly detection in attributed networks | Int. Conf. Inf. Knowl. Manage. | 2020 | _Li et al._ | [[Paper]()] |
| Fdgars: Fraudster detection via graph convolutional networks in online app review system | Int. Conf. World Wide Web | 2019 | _Wang et al._ | [[Paper]()] |
| Gcn-based user representation learning for unifying robust recommendation and fraudster detection | ACM SIGIR | 2020 | _Zhang et al._ | [[Paper]()] |
| Resgcn: Attention-based deep residual modeling for anomaly detection on attributed networks | Mach. Learn. | 2021 | _Pei et al._ | [[Paper]()] |

----------

#### Graph Attention Neural Network Based Techniques

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Anomalydae: Dual autoencoder for anomaly detection on attributed networks | IEEE Int. Conf. Acoustics Speech Signal Processing | 2020 | _Fan et al._ | [[Paper]()] |
| A semi-supervised graph attentive network for financial fraud detection | IEEE Int. Conf. Data Mining | 2019 | _Wang et al._ | [[Paper]()] |

----------

#### Generative Adversarial Neural Network Based Techniques

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Inductive anomaly detection on attributed networks | Int. Joint Conf. Artif. Intell. | 2020 | _Ding et al._ | [[Paper]()] |

----------

#### Reinforcement Learning Based Techniques
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Interactive anomaly detection on attributed networks | Int. Conf. Web Search Data Mining | 2019 | _Ding et al._ | [[Paper]()] |
| Selective network discovery via deep reinforcement learning on embedded spaces | Appl.Network Sci. | 2021 | _Morales et al._ | [[Paper]()] |

----------

#### Network Representation Based Techniques
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| A robust embedding method for anomaly detection on attributed networks | Int. Joint Conf. Neural Netw. | 2019 | _Zhang et al._ | [[Paper]()] |
| Enhancing graph neural network-based fraud detectors against camouflaged fraudsters | ACM Int. Conf. Inf. Knowl. Manage. | 2020 | _Dou et al._ | [[Paper]()] |
| Semi-supervised embedding in attributed networks with outliers | SIAM Int. Conf. Data Mining | 2018 | _Liang et al._ | [[Paper]()] |
| One-class graph neural networks for anomaly detection in attributed networks | Neural Comput. Appl. | 2021 | _Wang et al._ | [[Paper]()] |
| Error-bounded graph anomaly loss for gnns | ACM Int. Conf. Inf. Knowl. Manage. | 2021 | _Zhao et al._ | [[Paper]()] |
| Anomaly detection on attributed networks via contrastive self-supervised learning | IEEE Trans. Neural Networks Learn. Syst. | 2021 | _Liu et al._ | [[Paper]()] |
| Cross-domain graph anomaly detection | IEEE Trans. Neural Networks Learn. Syst. | 2021 | _Ding et al._ | [[Paper]()] |
| Fraudre: Fraud detection dual-resistant to graph inconsistency and imbalance | Pacific-Asia Conf. Knowl. Discov. Data Mining | 2021 | _Zhang et al._ | [[Paper]()] |
| Few-shot network anomaly detection via cross-network meta-learning | Web Conf. | 2021 | _Ding et al._ | [[Paper]()] |

----------

### Anomalous Node Detection on Dynamic Graphs
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Netwalk: A flexible deep embedding approach for anomaly detection in dynamic networks | ACM SIGKDD | 2018 | _Yu et al._ | [[Paper]()] |
| Anomaly detection in dynamic networks using multi-view time-series hypersphere learning | ACM Int. Conf. Inf. Knowl. Manage. | 2017 | _Teng et al._ | [[Paper]()] |
| One-class adversarial nets for fraud detection | AAAI Conf. Artif. Intell. | 2019 | _Zheng et al._ | [[Paper]()] |

----------

## Anomalous Edge Detection

| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Unified graph embedding-based anomalous edge detection | Int. Joint Conf. Neural Netw. | 2020 | _Ouyang et al._ | [[Paper]()] |
| Aane: Anomaly aware network embedding for anomalous link detection | IEEE Int. Conf. Data Mining | 2020 | _Duan et al._ | [[Paper]()] |
| efraudcom: An e-commerce fraud detection system via competitive graph neural networks | ACM Trans. Inf. Syst. | 2021 | _Zhang et al._ | [[Paper]()] |
| Netwalk: A flexible deep embedding approach for anomaly detection in dynamic networks | ACM SIGKDD | 2018 | _Yu et al._ | [[Paper]()] |
| Addgraph: Anomaly detection in dynamic graph using attention-based temporal gcn | Int. Joint Conf. Artif. Intell. | 2019 | _Zheng et al._ | [[Paper]()] |

----------

## Anomalous Sub-graph Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Deep structure learning for fraud detection | IEEE Int. Conf. Data Mining | 2018 | _Wang et al._ | [[Paper]()] |
| Fraudne: A joint embedding approach for fraud detection | Int. Joint Conf. Neural Netw. | 2018 | _Zheng et al._ | [[Paper]()] |

----------

## Anomalous Graph Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| User preference-aware fake news detection | arXiv | 2021 | _Dou et al._ | [[Paper](https://arxiv.org/abs/2104.12259)] |
| On using classification datasets to evaluate graph outlier detection: Peculiar observations and new insights | arXiv | 2021 | _Zheng et al._ | [[Paper](https://arxiv.org/abs/2012.12931)] |
| Deep into hypersphere: Robust and unsupervised anomaly discovery in dynamic networks | Int. Joint Conf. Artif. Intell. | 2018 | _Teng et al._ | [[Paper]()] |
| Glad-paw: Graph-based log anomaly detection by position aware weighted graph attention network | Pacific-Asia Conf. Knowl. Discov. Data Mining | 2021 | _Zheng et al._ | [[Paper]()] |

----------

## Published Algorithms and Models

| Model | Language | Platform | Graph | Code Repository |
| --- | -- | ---- | ---- | -----|
|Sedanspot | C++ | - | Dynamic Graph | https://www.github.com/dhivyaeswaran/sedanspot |
|AnomalyDAE | Python | Tensorflow | Dynamic Attribute Graph | https://github.com/haoyfan/AnomalyDAE |
|MADAN | Python| - |Static Attributed Graph | https://github.com/leoguti85/MADAN|
|PAICAN | Python | Tensorflow | Static Attributed Graph | http://www.kdd.in.tum.de/PAICAN/ |
| Changedar | Matlab | - |Dynamic Attributed Graph| https://bhooi.github.io/changedar/|
|ONE | Python |- |Static Plain Graph |https://github.com/sambaranban/ONE|
|DONE&AdONE | Python| Tensorflow |Static Attributed Graph |https://bit.ly/35A2xHs|
|SLICENDICE  |Python |- |Static Attributed Graph |http://github.com/hamedn/SliceNDice/|
|FRAUDRE | Python |Pytorch |Static Attributed Graph |https://github.com/FraudDetection/FRAUDRE|
|SemiGNN | Python |Tensorflow | Static Attributed Graph | https://github.com/safe-graph/DGFraud|
|CARE-GNN | Python |Pytorch| Static Attributed Graph |https://github.com/YingtongDou/CARE-GNN|
|GraphConsis | Python |Tensorflow |Static Attributed Graph |https://github.com/safe-graph/DGFraud|
|GLOD | Python |Pytorch |Static Attributed Graph |https://github.com/LingxiaoShawn/GLOD-Issues|
|GCAN | Python |Keras| Heterogeneous Graph |https://github.com/l852888/GCAN|
|HGATRD | Python |Pytorch |Heterogeneous Graph |https://github.com/201518018629031/HGATRD|
|GLAN | Python |Pytorch |Heterogeneous Graph |https://github.com/chunyuanY/RumorDetection|
|ANOMRANK | C++ |- |Dynamic Graph |https://github.com/minjiyoon/anomrank|
|DAGMM | Python |Pytorch |Dynamic Graph |https://github.com/danieltan07/dagmm|
|F-FADE | Python |Pytorch |Dynamic Graph |http://snap.stanford.edu/f-fade/|
|OCAN | Python |Tensorflow |Static Graph |https://github.com/PanpanZheng/OCAN|
|DevNet | Python |Tensorflow |Static Graph |https://github.com/GuansongPang/deviationnetwork|
|RDA | Python |Tensorflow |Static Graph |https://github.com/zc8340311/RobustAutoencoder|
|GAD | Python |Tensorflow |Static Graph |https://github.com/raghavchalapathy/gad|
|GEM | Python |- |Static Graph |https://github.com/safe-graph/DGFraud/tree/master/algorithms/GEM|
|eFraudCom | Python |Pytorch |Static Graph |https://github.com/GeZhangMQ/eFraudCom|
|MIDAS  |C++ |- |Dynamic Graph |https://github.com/Stream-AD/MIDAS|
|DeFrauder | Python |- |Static Graph |https://github.com/LCS2-IIITD/DeFrauder|
|DeepFD | Python| Pytorch |Bipartite Graph| https://github.com/JiaWu-Repository/DeepFDpyTorch|
|STS-NN | Python| Pytorch |Static Graph |https://github.com/JiaWu-Repository/STS-NN|
|UPFD | Python |Pytorch |Graph Database |https://github.com/safe-graph/GNN-FakeNews|
|DeepSphere | Python| Tensorflow |Dynamic Graph |https://github.com/picsolab/DeepSphere|
|OCGIN | Python |Pytorch |Graph Database |https://github.com/LingxiaoShawn/GLOD-Issues|
|Deep SAD | Python |Pytorch |Non Graph |https://github.com/lukasruff/Deep-SAD-PyTorch|
|DATE | Python |Pytorch |Non Graph |https://github.com/Roytsai27/Dual-Attentive-Treeaware-Embedding|

----------

## Datasets
### Citation/Co-authorship Networks
- Citeseer, Cora, Pubmed https://linqs.soe.ucsc.edu/data
- DBLP http://snap.stanford.edu/data/com-DBLP.html, http://www.informatik.uni-trier.de/ ̃ley/db/
- ACM http://www.arnetminer.org/open-academic-graph
### Social Networks
- Enron http://odds.cs.stonybrook.edu/#table2
- UCI Message http://archive.ics.uci.edu/ml
- Google+ https://wangbinghui.net/dataset.html
- Twitter Sybil https://wangbinghui.net/dataset.html
- Twitter World-Cup2014 http://shebuti.com/SelectiveAnomalyEnsemble/
- Twitter Security2014 http://shebuti.com/SelectiveAnomalyEnsemble/
- Reality Mining http://shebuti.com/SelectiveAnomalyEnsemble/
- NYTNews http://shebuti.com/SelectiveAnomalyEnsemble/
- Politifact https://github.com/safe-graph/GNN-FakeNews
- Gossipcop https://github.com/safe-graph/GNN-FakeNews
### Co-purchasing Networks
- Disney Calls https://www.ipd.kit.edu/mitarbeiter/muellere/consub/
- Amazon-v1 https://www.ipd.kit.edu/mitarbeiter/muellere/consub/
- Amazon-v2 https://github.com/dmlc/dgl/blob/master/python/dgl/data/fraud.py
- Elliptic https://www.kaggle.com/ellipticco/elliptic-data-set
- Yelp https://github.com/dmlc/dgl/blob/master/python/dgl/data/fraud.py
### Transportation Networks
- New York City Taxi http://www.nyc.gov/html/tlc/html/about/triprecorddata.shtml

----------

## Tools
- Gephi https://gephi.org/
- Pajek http://mrvar.fdv.uni-lj.si/pajek/
- LFR https://www.santofortunato.net/resources

----------
**Disclaimer**

If you have any questions, please feel free to contact us.
Emails: <u>xiaoxiao.ma2@hdr.mq.edu.au</u>, <u>jia.wu@mq.edu.au</u>

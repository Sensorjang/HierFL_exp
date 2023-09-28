<div align="center">
<h1 align="center">FARA</h1>
TMC实验攻关

![GitHub](https://img.shields.io/github/license/Sensorjang/HierFL_exp)
![Language](https://img.shields.io/badge/Language-Python-blue)

</div>

源码来自论文《Client-Edge-Cloud Hierarchical Federated Learning》https://paperswithcode.com/paper/edge-assisted-hierarchical-federated-learning
数据集为mnist及cifar-10，模型包括cnn和resnet。


内容介绍：该论文将模型聚合分为两个阶段：边缘聚合和云聚合。

设有一个云服务器，L个边缘服务器，每个边缘服务器都有不相交的设备集D_n.

D_n中的设备进行本地更新，上传模型至边缘服务器n，边缘服务器聚合D_n中的设备提交的参数，并发回进行下一轮更新；完成一定数量的边缘聚合后，L个边缘服务器上传模型至云服务器进行一轮全局聚合。（以节省通信开销）。

设每轮边缘聚合需要经过K1轮本地更新，每轮云聚合（epoch）需要经过K2轮边缘聚合，故每轮云聚合共需K1K2轮本地更新。


修改要求：分为简化版本和noniid版本。

简化版本：需要能够调整
边缘服务器数量L，
每个边缘服务器的设备个数|D_n|，
每个设备的本地样本量s（各个设备样本量一致，从训练集划分），
以及轮次数K1、K2和迭代次数（云聚合次数）。
输出结果用三个数组打印（云聚合轮次数，全局精度，平均损失）。


Noniid版本：需要能够调整
边缘服务器数量L，
每个设备的本地样本量s（随机分布（可以是均匀分布），总数固定），
每轮边缘聚合中，每个边缘服务器的设备集D_n， 由论文和baseline的算法决定，
以及轮次数K1、K2和迭代次数（云聚合次数）。
输出结果用三个数组打印（云聚合轮次数，全局精度，平均损失）。

 
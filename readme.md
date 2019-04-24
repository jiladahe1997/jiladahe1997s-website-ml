[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>

# 介绍

这是我的毕业设计题目：基于人脸图像进行年龄预测。 

其原理是基于机器学习（machine learning），具体原理请见本文最后 - 原理介绍

文件比较乱：待整理。<br/>
数据处理: mat.py、main_eager.py、save_to_tfrecord.py <br/>
训练模型：model_from_tfrecord.py <br/>

如果你也想训练自己的神经网络，或者也想学习机器学习，教程推荐：[todo]



<br/>

# 数据集、机器学习框架、web框架等

1.数据集使用的是IMDB-WIKI数据集：https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}

2.机器学习框架使用的是TensorFlow

3.web框架使用的是django

4.训练使用的google colab

# 使用示例

图片太大了传不了 点这里 http://120.78.151.148/ml 可以看

# 踩过的坑


1.CPU训练图像30S一张，温度90度，我真的怕我的神船Z7刚不住。启用GPU：过程比较复杂，但是参照https://www.tensorflow.org/install/gpu，可以顺利完成，**务必注意CUDA 必须为10.0版本，安装了个10.1版本用不了佛了** （启用后0.7s一张图片）

# todo:

√ 1.修改metric
就用MAL

√ 2.调整batch和steps_per_epoch 
batch指一次训练的图片张数，steps_per_epoch指训练几个batch

√3.test_data

√ 4.CPU 90度刚不住，启用GPU
见上面

√(高优先度)5.现在训练20个epochs后absolute loss固定在12-13，需要优化。→ 6.修改回归输出→分段回归

√(高优先度)7.分批运行，避免显存爆炸

√ 8.训练完后保存
调用model.save


# 原理介绍

假设输出结果和输入之间存在某种关系式，例如z=w1x+w2y，其中x和y是机器学习的输入，z是机器学习的输出，w1和w2都是未知的。（假设正确的关系式是：z=x+y）

现在有n组xyz，x=1,2,3 y=1,2,3 z=2,4,6

第一步：初始化一个w1 w2，假设w1=1 w2=2, 则现在的关系式是 z=x+2y

第二步：输入x=1 y=1 z=2，那么现在得到的预测结果Zpredict = zp = 3。其离正确值3还相差1。接下来就要调整w1 w2减少这个误差。

第三部：首先误差表达式可以表示为，即问题转换为求这个函数的极小值

<img src="./readme_1.png"  width=200 />

第四步：首先对w1求导：

<img src="./readme_2.png"  width=200 />

再对w2求导

<img src="./readme_3.png"  width=200 />

第五步：然后沿着两个方向分别取更小的 w1 w2值，计算新的z值（梯度下降：https://zh.wikipedia.org/zh-hk/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95）

第六步：重复第五步，直到z的值不下降为止

第七步：找到新的w1，w2值，重新代入第二组数据重复1-7步
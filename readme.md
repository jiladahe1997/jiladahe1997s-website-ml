# 踩过的坑

1.CPU训练图像30S一张，温度90度，我真的怕我的神船Z7刚不住。启用GPU：过程比较复杂，但是参照https://www.tensorflow.org/install/gpu，可以顺利完成，**务必注意CUDA 必须为10.0版本，安装了个10.1版本用不了佛了** （启用后0.7s一张图片）

# todo:

1.修改metric

√ 2.调整batch和steps_per_epoch 
batch指一次训练的图片张数，steps_per_epoch指训练几个batch

3.调整test_data

√ 4.CPU 90度刚不住，启用GPU
见上面

5.现在训练20个epochs后absolute loss固定在12-13，需要优化
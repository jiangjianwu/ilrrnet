# ILRRNet (Indoor Layout Reasonability Recognition Network)

## 1.安装依赖库

```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.数据集下载

点击此链接[Indoor Rational Dataset](https://gluteducn-my.sharepoint.com/:u:/g/personal/fengbuxi_glut_edu_cn/EVjC4ESt-a9CnfBe48gT7wUBxRL7hJNdoqiHUTBfNzyoMw?e=xwrRlf)下载数据集。

密码：fengbuxi

## 3.训练
```bash
python train.py
```

## 4.验证
```bash
python eval.py
```

## 5.融合地图构建
```bash
python cognition_map.py
```

## 联系方式

    + Email: fengbuxi@glut.edu.cn

    + WebSite: https://www.jaingjianwu.cn/

## 引用

```
@article{ICENet,
  title={Construction of Indoor Element-Level Layout Rationality Maps Driven by Semantic Cognition}, 
  author={Jianwu Jiang and Yuhao Xiang and Jingwen Li},
  journal={},
  year={2024}
}
```

## 鸣谢

+ 本模型基于[PSPNet](https://arxiv.org/pdf/1612.01105)构建。

+ 本模型基于开源架构[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git)构建。

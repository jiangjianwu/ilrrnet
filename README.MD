# ILRRNet (Indoor Layout Reasonability Recognition Network)

## 1. Installation of dependent libraries

```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.Dataset Download

Click on [Indoor Rational Dataset](https://gluteducn-my.sharepoint.com/:u:/g/personal/fengbuxi_glut_edu_cn/EVjC4ESt-a9CnfBe48gT7wUBxRL7hJNdoqiHUTBfNzyoMw?e=xwrRlf) to download the dataset.


password: fengbuxi

## 3.Train

```bash
python train.py
```

## 4.Eval

```bash
python eval.py
```

## 5.Map Create

```bash
python cognition_map.py
```

## Connection

    + Email: fengbuxi@glut.edu.cn

    + WebSite: https://www.jaingjianwu.cn/

## Citation

```
@article{ICENet,
  title={Construction of Indoor Element-Level Layout Rationality Maps Driven by Semantic Cognition}, 
  author={Jianwu Jiang and Yuhao Xiang and Jingwen Li},
  journal={},
  year={2024}
}
```


## Acknowledgements

+ This model is constructed based on [PSPNet](https://arxiv.org/pdf/1612.01105).

+ This model is built on the open source framework [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git).
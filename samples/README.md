# 样本截图目录

这个目录存放**用于测试和演示的骑行 App 截图**。

## 为什么不入库

默认 `.gitignore` 排除了 `samples/*.jpg`。原因是这类截图包含**个人位置信息**：
起终点周边的真实住址区域、骑行路线、常去的地标。放进公开仓库等于公开一段
位置历史。

`.gitignore` 里对 `docs/*.png`（轨迹可视化）和 `trajectory*.geojson`
（原始 GPS 轨迹）做了同样处理。

## 怎么用

把自己的截图放进这个目录即可（文件名可任意，测试会按路径查找）：

```
samples/
└── meituan_ride_shanghai.jpg    ← 测试用样本
    meituan_ride_2.jpg           ← 可选，第二张样本
```

然后：

```bash
pip install -r requirements.txt
python tests/test_e2e.py                       # 跑验收测试
python -m app.cli samples/meituan_ride_shanghai.jpg --out out/
```

测试在样本缺失时会打印「样本缺失」并以非零码退出，**不会崩溃**。

## 如果你愿意公开自己的样本

去掉 `.gitignore` 里对应的两行即可：

```
# samples/*.jpg
# docs/*.png
# trajectory*.geojson
```

或者更省事——用 `git add -f` 强制添加单个文件：

```bash
git add -f samples/meituan_ride_shanghai.jpg
```

## 合成样本

如果需要一个不含隐私的演示样本，可以合成一张：任意底图 + 人工绘制的灰色
折线 + 起终点徽章。轨迹提取阶段不依赖真实数据，但**地理编码阶段需要真实
POI 标注**才能配准，所以纯合成图只能测前半段流水线。

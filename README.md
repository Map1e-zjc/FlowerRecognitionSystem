# AI 花卉识别系统（Flower Recognition System）

> **项目状态：实施中 —— M1（环境与数据管道）✅ 完成 · M2（模型训练与对比）✅ 完成**
> 下一步：M3 后端 API（认证 / 识别 / 百科 / 历史 / 模型对比）。
> 进度与逐条判据见 [`docs/04-实施步骤与里程碑.md`](docs/04-实施步骤与里程碑.md)。

---

## 一、一句话描述

基于深度迁移学习的 Web 花卉识别系统：用户在浏览器上传花卉照片，系统返回 **Top-5 候选种类与置信度**，并提供花卉百科、识别历史、模型对比可视化，配套完整训练脚本与实验报告。

## 二、当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 项目准备：选题澄清、需求冻结、技术方案、实施步骤 | ✅ 完成 |
| M1 | 环境搭建 + 数据管道（8 步） | ✅ 完成 |
| M2 | 模型训练、对比与评估（8 步，M2-7 按计划跳过） | ✅ 完成 |
| M3 | 后端 API（认证 / 识别 / 百科 / 历史 / 模型对比） | ⏸ 下一步 |
| M4 | 前端 Vue3（识别 / 百科 / 历史 / 可视化） | ⏸ 未开始 |
| M5 | 联调、一键启动、报告与答辩材料 | ⏸ 未开始 |

## 三、M2 模型对比结果（测试集 6149 张，官方 P1 划分）

| 模型 | 预训练 | 参数 | Test Top-1 | Test Top-5 | Macro-F1 | 单张推理 | 训练耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ViT-B/16**（最优） | AugReg IN-21k→1k | 85.9 M | **99.59%** | 99.79% | 99.50% | 10.14 ms | 16.0 min |
| ConvNeXt-Tiny | ImageNet-22k→1k | 27.9 M | 99.46% | **99.80%** | 99.41% | **9.46 ms** | **11.8 min** |
| ResNet-50 | ImageNet-1k V2 | 23.7 M | 94.26% | 98.94% | 94.13% | 10.26 ms | 24.3 min |
| EfficientNet-B0 | NoisyStudent JFT→1k | 4.1 M | 91.87% | 98.10% | 91.32% | 13.67 ms | 25.3 min |

**验收目标达成**：最优模型 Top-1 **99.59% ≥ 95%** ✅、Top-5 **99.79% ≥ 99%** ✅。

核心结论：**预训练强度比网络架构更能决定小样本细粒度任务的上限**
（22k/21k 预训练的两个模型 ≥99.4%，仅 1k 预训练的相差 5 个百分点以上）。
详细错误分析见 [`docs/08-错误分析.md`](docs/08-错误分析.md)。

## 四、已确认的关键决策（摘要）

| 项 | 结论 |
| --- | --- |
| 交付形态 | Web 系统，浏览器上传图片识别 |
| 技术栈 | PyTorch + FastAPI + Vue3 |
| 数据集 | Oxford Flowers-102（102 类 / 8189 张），经 hf-mirror 镜像获取 |
| 功能范围 | Top-5 识别、花卉百科、识别历史、用户登录注册、模型对比可视化、训练脚本 + 实验报告 |
| 运行环境 | 本机 RTX 4060 Laptop (8 GB) 训练 + 本机运行演示 |
| 环境隔离 | D 盘 `.venv` + CUDA 版 PyTorch `2.14.0+cu126`（保护 C 盘） |
| 用户体系 | SQLite + JWT，单角色普通用户 |
| 部署方式 | 不用 Docker，本机一键启动脚本 |

完整理由（含执行中变更的 ADR-011~015）见 [`docs/00-项目决策记录.md`](docs/00-项目决策记录.md)。

## 五、文档导航

| 文档 | 内容 |
| --- | --- |
| [`docs/00-项目决策记录.md`](docs/00-项目决策记录.md) | ADR 决策记录，含每项结论的理由、被否决方案与执行中变更 |
| [`docs/01-需求与范围.md`](docs/01-需求与范围.md) | 用户角色、功能需求 FR、非功能需求 NFR、范围边界、交付物 |
| [`docs/02-技术架构与接口契约.md`](docs/02-技术架构与接口契约.md) | 架构图、选型表、目录结构、REST API 契约、数据库 schema |
| [`docs/03-数据集与模型方案.md`](docs/03-数据集与模型方案.md) | Flowers-102 划分、数据质量核查、模型候选与**实测结果**、超参、评估协议 |
| [`docs/04-实施步骤与里程碑.md`](docs/04-实施步骤与里程碑.md) | **核心**：M1–M5 逐条实施步骤、命令、产出物、完成判据、执行结果 |
| [`docs/05-环境准备与依赖清单.md`](docs/05-环境准备与依赖清单.md) | 本机环境实测、依赖清单、磁盘/缓存重定向、性能基线、踩坑规则 |
| [`docs/06-风险登记与验收标准.md`](docs/06-风险登记与验收标准.md) | 风险登记册与应对方案、最终验收清单 |
| [`docs/08-错误分析.md`](docs/08-错误分析.md) | 四模型混淆矩阵分析、系统性混淆发现、改进方向 |

## 六、如何跑起来（当前可用部分）

```powershell
# 1) 激活环境（自动设置缓存重定向与 HF 镜像）
. .\scripts\activate.ps1

# 2) 数据准备（已就位，如需重建）
.\scripts\download_data.ps1                 # 下载 + 校验（hf-mirror 源）
python -m ml.eda                            # 数据探索图表

# 3) 训练（单个 / 全部）
.\scripts\train.ps1 -Model vit_b16
.\scripts\train_all.ps1                     # 4 个模型串行 + 自动评估汇总

# 4) 评估与指标
python -m ml.evaluate --weights ml\outputs\checkpoints\vit_b16_best.pth --split test --no-progress
python -m ml.export_metrics                 # → ml\outputs\metrics.json

# 5) 花卉百科数据
python -m ml.build_flowers_json --export-thumbs   # → backend\app\data\flowers.json
```

> 后台/脚本中运行训练请加 `--no-progress`，否则 tqdm 会把日志刷爆（原因见 docs/05 §7.2）。

## 七、当前仓库内容

```
README.md
docs/                    8 篇计划与执行文档
ml/                      数据管道、训练、评估、指标导出、百科构建（9 个模块）
ml/configs/              4 个模型的 YAML 配置
ml/outputs/              权重、指标、图表（权重与日志不入库，图表与 metrics.json 入库）
backend/app/data/        flowers.json（102 类中文百科）
backend/app/static/flowers/  102 张百科缩略图
backend/requirements.txt 冻结的依赖版本（75 项）
scripts/                 激活、数据下载、单模型训练、批量训练
data/  .cache/  .venv/   数据集、缓存、虚拟环境（均不入库）
```

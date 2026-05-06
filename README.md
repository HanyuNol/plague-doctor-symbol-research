# COVID-19 & Plague Doctor Image Analysis Pipeline
 
> 基于多模态大模型的疫病防控医护形象视觉符号分析工具
 
本仓库是论文 **《疫病防控中医护形象符号建构的比较研究——基于文化符号学视角》** 的配套代码与数据。研究通过爬虫采集黑死病（Plague Doctor）与新冠疫情（COVID-19）主题图像，利用 Qwen2.5-VL 多模态大模型进行自动标注，并从文化符号学视角对两个历史时期的医护视觉符号进行系统比较分析。
 
---
 
## 研究概述
 
| 项目 | 说明 |
|------|------|
| **研究对象** | 14世纪黑死病中的"鸟嘴医生"（Plague Doctor）与2020年新冠疫情中的"大白"（防护服医护） |
| **理论框架** | 文化符号学（Lotman 符号圈、Eco 无限衍义、Moscovici 社会表征） |
| **核心发现** | 鸟嘴医生实际出现频率仅 1.6%（人工复核后），已发生"符号漂移"；大白出现频率 37.1%，呈现强"符号绑定" |
| **图像总量** | 1,724 张（Bridgeman 500 + Pexels 480 + Pinterest 474） |
| **标注模型** | Qwen2.5-VL-3B-Instruct（4-bit 量化） |
 
---
 
## 项目结构
 
```
covid_image_pipeline/
├── .env                          # 环境变量配置（API Key、模型参数等）
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
│
├── run_all.py                    # 一键运行 COVID-19 图像全流程
├── run_blackdeath.py             # 一键运行黑死病（Bridgeman）全流程
├── run_pinterest.py              # 一键运行 Pinterest 全流程
├── test.py                       # 测试脚本
│
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── config.py                 # 全局配置（API Key、模型参数、路径等）
│   ├── utils.py                  # 工具函数（JSON 解析、目录创建等）
│   │
│   ├── fetch_unsplash.py         # Pexels API 图像采集
│   ├── fetch_bridgeman.py        # Bridgeman Education 艺术图像采集
│   ├── fetch_pinterest.py        # Pinterest 社交平台图像采集
│   │
│   ├── classify_images.py        # COVID-19 图像标注（Qwen2.5-VL）
│   ├── classify_bridgeman.py     # 黑死病艺术图像标注
│   ├── classify_pinterest.py     # Pinterest 图像标注
│   │
│   ├── stats_report.py           # COVID-19 数据统计分析
│   ├── stats_bridgeman.py        # 黑死病数据统计分析
│   └── stats_pinterest.py        # Pinterest 数据统计分析
│
├── data/                         # 图像数据
│
└── models/                       # 本地模型缓存（Qwen2.5-VL）
```
 
---
 
## 环境要求
 
- **Python**: 3.9+
- **GPU**: NVIDIA GPU（建议显存 ≥ 8GB，用于模型推理）
- **CUDA**: 11.8+
 
### 主要依赖
 
```
torch>=2.0
transformers>=4.40
qwen-vl-utils
accelerate
datasets
pandas
tqdm
requests
selenium          # Pinterest 爬虫
webdriver-manager # ChromeDriver 管理
python-dotenv
Pillow
matplotlib
```
 
---
 
## 快速开始
 
### 1. 克隆仓库
 
```bash
git clone https://github.com/<your-username>/covid_image_pipeline.git
cd covid_image_pipeline
```
 
### 2. 安装依赖
 
```bash
pip install -r requirements.txt
```
 
### 3. 配置环境变量
 
复制 `.env.example` 为 `.env`，填入你的配置：
 
```bash
cp .env.example .env
```
 
`.env` 文件内容：
 
```env
# Pexels API Key（从 https://www.pexels.com/api/ 获取）
UNSPLASH_ACCESS_KEY=your_pexels_api_key_here
 
# 搜索关键词
SEARCH_QUERY=covid 19
PER_PAGE=30
MAX_PAGES=3
 
# 模型配置
MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
TEMPERATURE=0.1
MAX_NEW_TOKENS=256
```
 
### 4. 运行
 
```bash
# COVID-19 图像全流程（采集 → 标注 → 统计）
python run_all.py
 
# 黑死病艺术图像全流程（Bridgeman）
python run_blackdeath.py
 
# Pinterest 社交图像全流程
python run_pinterest.py
```
 
---
 
## 数据采集
 
### 数据源
 
| 数据源 | 关键词 | 图像数量 | 说明 |
|--------|--------|----------|------|
| **Bridgeman Education** | `plague doctor`, `black death` | 500 | 中世纪及近代欧洲艺术作品数据库 |
| **Pexels API** | `covid 19` | 480 | 免费图库平台，通过 API 批量获取 |
| **Pinterest** | `plague doctor`, `black death` | 270 + 204 | 社交视觉平台，反映公众视觉认知 |
 
### 采集脚本
 
- `src/fetch_unsplash.py` — 通过 Pexels REST API 按关键词搜索并下载图像
- `src/fetch_bridgeman.py` — 从 Bridgeman Education 数据库检索艺术图像
- `src/fetch_pinterest.py` — 基于 Selenium 的 Pinterest 滚动加载与图像下载
 
---
 
## 图像标注
 
### 模型
 
使用 **Qwen2.5-VL-3B-Instruct**（4-bit 量化），通过 `transformers` 库本地部署。
 
```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
```
 
### 提示词设计
 
针对三类数据集设计了差异化的标注提示词：
 
#### 黑死病艺术图像提示词（`PROMPT_BLACKDEATH`）
 
```
你是一个严谨的历史图像分析专家。
请分析这张图片，并严格输出 JSON，不要输出任何额外说明。
 
任务1：判断是否有人物
- 如果图片中存在至少一个人（不论古今），则 "has_people": 1
- 否则 "has_people": 0
 
任务2：判断是否有"鸟嘴医生"形象（仅当 has_people=1 时认真判断）
鸟嘴医生（Plague Doctor）特征：
- 穿着黑色或深色长袍
- 戴着鸟嘴形状的面具（喙状突起）
- 可能配有宽檐帽、手套、手杖
如果画面中出现符合上述特征的医生形象，则 "has_plague_doctor": 1，否则为 0。
若 has_people=0，则此字段设为 0。
 
任务3：标签提取
从以下集合中选取相关标签，可补充少量其他标签：
["plague_doctor","beak_mask","medieval","costume","historical","physician",
 "black_death","mask","robe","staff","crowd","indoor","outdoor"]
 
输出格式：
{
  "has_people": 0/1,
  "has_plague_doctor": 0/1,
  "confidence": 0.95,
  "tags": ["tag1","tag2",...],
  "reason": "简短判断理由"
}
注意：只能输出 JSON，不要 markdown。
```
 
#### Pinterest 图像提示词（`PROMPT_PINTEREST`）
 
在黑死病提示词基础上增加了"死神"（Grim Reaper）判断任务：
 
```
任务3：判断是否有"死神"形象（仅当 has_people=1 时认真判断）
死神特征：骷髅骨架、身披黑袍、手持镰刀，或明显的死亡象征（如骷髅头、骨镰等）。
如果出现上述特征，则 "has_grim_reaper": 1，否则为 0。若 has_people=0，此字段为 0。
 
任务4：标签提取
从以下集合中选取相关标签，可补充少量其他标签：
["plague_doctor","grim_reaper","skeleton","scythe","robe","mask","beak_mask",
 "medieval","historical","crowd","indoor","outdoor"]
```
 
#### 新冠疫情图像提示词（COVID-19）
 
```
你是一个严谨的公共卫生图像分析助手。
请分析这张图片，并严格输出 JSON，不要输出任何额外说明。
 
任务1：判断是否有人
- 如果图片中存在至少一个人（不论是否医务人员），则 "has_people": 1
- 否则 "has_people": 0
 
任务2：判断是否有穿防护服的医务人员（仅当 has_people=1 时认真判断）
- 如果画面中出现医护人员、医院工作人员、防疫人员，并且其穿着明显防护服、
  隔离服、防护面罩、面屏、全套医用防护装备(PPE)，
  则 "has_protective_medical_staff": 1
- 否则为 0
- 若 has_people=0，则此字段可设为 0
 
任务3：标签提取
从图像中提取若干标签。优先从以下集合中选择，允许少量补充：
["doctor","nurse","medical_staff","ppe","protective_suit","mask","face_shield",
 "gloves","hospital","clinic","patient","ambulance","testing","vaccination",
 "microscope","laboratory","virus","covid_sign","crowd","indoor","outdoor"]
 
输出格式必须是：
{
  "has_people": 0 或 1,
  "has_protective_medical_staff": 0 或 1,
  "confidence": 0.95,
  "tags": ["tag1","tag2",...],
  "reason": "简短判断理由"
}
注意：
1. 只能输出 JSON
2. 不要输出 markdown
3. confidence 取值范围 0-1
```
 
### 模型参数
 
| 参数 | 值 | 说明 |
|------|-----|------|
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-3B-Instruct` | 模型标识 |
| `TEMPERATURE` | `0.1` | 低温度以减少随机性 |
| `MAX_NEW_TOKENS` | `256` | 生成最大令牌数 |
| 量化 | 4-bit (bitsandbytes) | 降低显存占用 |
| 平均置信度 | 0.950 | 模型输出质量指标 |
 
### 标注输出格式
 
每张图像输出一个 JSON 对象，包含：
 
```json
{
  "has_people": 1,
  "has_plague_doctor": 0,
  "has_grim_reaper": 1,
  "has_protective_medical_staff": 0,
  "confidence": 0.95,
  "tags": ["skeleton", "scythe", "robe", "mask", "grim_reaper"],
  "reason": "图像中有一个骷髅骨架，身穿黑袍，手持镰刀，符合死神的特征。"
}
```
 
---
 
## 人工复核说明
 
模型自动标注后，需进行人工复核。主要发现：
 
### Bridgeman 数据集误判问题
 
模型原始标注鸟嘴医生出现率为 **16.0%**（76/476），人工复核后修正为 **1.6%**（8/476）。
 
**误判类型：**
 
| 类型 | 描述 | 示例 |
|------|------|------|
| 长袍误判 | 将身穿深色长袍的修士/神父误判为鸟嘴医生 | 宗教绘画中的主教形象 |
| 模糊背影误判 | 远景或背影人物，面部不可辨识 | 人群远景中的模糊人物 |
| 年代混淆误判 | 17-19世纪民俗画中穿普通外套的人物 | 非鸟嘴面具的普通长外套 |
 
**误判原因分析：**
 
1. **提示词偏差（Prompt Bias）**：引导性提问激活模型先验知识，模糊案例中倾向肯定判断
2. **训练数据偏差（Training Data Bias）**：长袍与鸟嘴医生在训练数据中高度共现，模型过度泛化
3. **文化知识缺失（Cultural Knowledge Gap）**：模型对中世纪服饰体系、宗教绘画传统的理解有限
 
---
 
## 核心数据统计
 
### 各数据集鸟嘴医生出现频率（人工复核后）
 
| 数据集 | 总图像 | 含人物 | 鸟嘴医生 | 频率 |
|--------|--------|--------|----------|------|
| Bridgeman (黑死病艺术) | 500 | 476 | 8 | **1.6%** |
| Pinterest "black death" | 204 | 165 | 10 | **4.9%** |
| Pinterest "plague doctor" | 270 | 252 | 176 | **65.2%** |
| Pexels (新冠图像) | 480 | 167 | — | — |
 
### 新冠疫情图像防护服医护出现频率
 
| 数据集 | 含人物 | 防护服医护 | 频率 |
|--------|--------|------------|------|
| Pexels (新冠图像) | 167 | 62 | **37.1%** |
 
---
 
## 关联论文
 
**朱贺.** 疫病防控中医护形象符号建构的比较研究——基于文化符号学视角[D]. 清华大学, 2025.
 
**核心参考文献：**
 
- Eco, U. (1976). *A Theory of Semiotics*. Indiana University Press.
- Lotman, Y. M. (1990). *Universe of the Mind: A Semiotic Theory of Culture*. Indiana University Press.
- Moscovici, S. (1984). The phenomenon of social representations. In *Social Representations*. Cambridge University Press.
- McLuhan, M. (1964). *Understanding Media: The Extensions of Man*. McGraw-Hill.
- Kress, G. & van Leeuwen, T. (2006). *Reading Images: The Grammar of Visual Design* (2nd ed.). Routledge.
- Wang, P. et al. (2024). Qwen2.5-VL Technical Report. arXiv:2502.13923.
- Zuboff, S. (2019). *The Age of Surveillance Capitalism*. PublicAffairs.
 
---
 
## 许可证
 
本项目代码采用 [MIT License](LICENSE) 开源。
 
**数据使用声明：**
 
- Bridgeman Education 图像受版权保护，仅供学术研究使用，请勿二次分发
- Pexels 图像遵循 [Pexels License](https://www.pexels.com/license/)（免费使用，需署名）
- Pinterest 图像版权归原作者所有，仅供学术研究引用
 
---
 
## 致谢
 
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 多模态视觉语言模型
- [Bridgeman Education](https://www.bridgemanimages.com/) — 艺术图像数据库
- [Pexels API](https://www.pexels.com/api/) — 免费图库接口
- [Pinterest](https://www.pinterest.com/) — 社交视觉平台

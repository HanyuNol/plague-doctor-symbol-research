\# COVID-19 图像抓取与分析项目



\## 项目功能

本项目实现如下完整流程：



1\. 从 Unsplash 搜索 COVID-19 相关图像

2\. 下载图片到本地

3\. 使用视觉大模型对图片进行分析

4\. 判断图片是否包含“身穿防护服的医务人员”

5\. 自动提取图像标签

6\. 输出统计分析结果



\## 项目目录

```text

covid\_image\_pipeline/

├─ .env

├─ requirements.txt

├─ README.md

├─ run\_all.py

├─ data/

│  ├─ raw/

│  ├─ meta/

│  ├─ outputs/

│  └─ logs/

└─ src/

&#x20;  ├─ config.py

&#x20;  ├─ utils.py

&#x20;  ├─ fetch\_unsplash.py

&#x20;  ├─ classify\_images.py

&#x20;  └─ stats\_report.py


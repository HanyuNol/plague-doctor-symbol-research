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

import json
from tqdm import tqdm
from datasets import Dataset
import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_valid_json(text):
    """修复的JSON提取函数 - 避免递归正则问题"""
    try:
        # 方法1: 直接查找第一个完整的JSON对象
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        stack = []
        json_str = ""
        
        for i in range(start_idx, len(text)):
            char = text[i]
            json_str += char
            
            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:  # 栈为空，找到完整JSON
                        try:
                            # 尝试解析验证
                            parsed = json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            # 继续寻找下一个
                            json_str = ""
                            continue
                else:
                    json_str = ""
                    
        return None
    except Exception as e:
        logger.warning(f"JSON查找失败: {e}")
        return None


def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    prompts, gold_data_list = [], []
    for item in tqdm(data, desc="加载数据"):
        prompt = build_scoring_prompt(item)
        prompts.append(prompt)
        gold_data_list.append(item)
    return prompts, gold_data_list

def build_scoring_prompt(item):
    
    #构建one shot
    few_shot_examples =[
        {
            "question": "计算等差数列的前n项和，已知首项a₁=2，公差d=3，项数n=5",
            "reference": "使用公式 Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40",
            "analysis": "正确应用等差数列求和公式，计算准确",
            "total": 2,
            "student_answer": [
                {"response": "Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40"}
            ],
            "output": {
                "total": 2,
                "steps": [
                    {
                        "response": "Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40", 
                        "step_score": 2, 
                        "errors": ["步骤正确"]
                    }
                ],
                "manual_label": 2
            }
        }
    ]

    few_shot_meta = {
        "prefix": "[参考示例]",
        "suffix": "以上为参考样例和输出格式，请参考这些样例的评分标准对下面的真实材料进行评估"
    }

    few_shot_blocks = []
    for i, example in enumerate(few_shot_examples, 1):
        block = (
            f"示例{i}:\n"
            f"- 试题内容：{example['question']}\n"
            f"- 题目分值：{example['total']}\n"
            f"- 标准答案：{example['reference']}\n"
            f"- 解析说明：{example['analysis']}\n"
            f"- 学生作答：{json.dumps(example['student_answer'], ensure_ascii=False, indent=2)}\n"
            f"输出: {json.dumps(example['output'], ensure_ascii=False, indent=2)}"
        )
        few_shot_blocks.append(block)

    few_shot_block = f"{few_shot_meta['prefix']}\n" + "\n\n".join(few_shot_blocks) + f"\n{few_shot_meta['suffix']}"

    # 指令部分
    instruction = """你是一名专业的数学阅卷教师。请严格按照下列 JSON 结构输出评分结果，不要添加任何多余字段或解释性文字：

{
  "total": <INT>,
  "steps": [
    {
      "response": "学生回答原文",
      "step_score": <INT>,
      "errors": ["<TAG1>", "<TAG2>"]
    }
  ],
  "manual_label": <INT>
}

评分规则：
1) step_score 取值说明：
   - 0: 步骤完全错误或无意义
   - 1: 步骤基本正确但有瑕疵
   - 2: 步骤完全正确且关键

2) errors 错误类型枚举（可多选）：
   ["步骤正确", "计算错误", "公式使用错误或遗漏", "答案与过程不符", 
    "解题思路混乱", "不相关作答", "笔误", "抄写错误", 
    "缺少关键步骤", "推理跳步", "概念理解错误"]

3) 输出要求：
   - 严格保持 steps 数组长度与学生作答完全一致
   - response 字段必须原样保留学生回答内容
   - 仅替换 <INT> 为整数分数，<TAG> 为错误类型字符串
   - 输出必须是有效的 JSON 格式
   - 不要包含题目内容、标准答案或任何解释性文字
   - 总分(total)和最终分(manual_label)应该一致
"""

    # 当前评分任务 - 适配你的数据结构
    student_answers = []
    for step in item["steps"]:
        student_answers.append({
            "response": step["response"]
        })

    current_block = (
        f"【当前评分任务】\n"
        f"- 题目分值：{item['total']}\n"
        f"- 标准答案：{item['reference']}\n"
        f"- 解析说明：{item.get('analysis', '无')}\n"
        f"- 学生作答：{json.dumps(student_answers, ensure_ascii=False, indent=2)}\n"
        f"输出:"
    )

    # 组合完整prompt
    full_prompt = f"{few_shot_block}\n\n{instruction}\n\n{current_block}"
    return full_prompt

#接下来开始创建训练数据集
def create_train_dataset(prompts, tokenizer):
    processed_data = []
    for prompt in tqdm(prompts, desc="处理数据"):
        encoding = tokenizer(
            prompt, 
            truncation=True,
            padding="max_length", 
            max_length=512
            )
        processed_data.append({
            "prompt": prompt,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
        })

    return Dataset.from_list(processed_data)


    
    
import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# ✅ 修复的JSON解析函数
# ===========================================================
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

# ===========================================================
# ✅ 多任务奖励模型
# ===========================================================
class MultiTaskRewardModel:
    """多任务奖励模型 - 基于大语言模型的判断"""
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载奖励模型
        logger.info(f"加载奖励模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("奖励模型加载完成")
    
    def compute_reward(self, prompt, generated_text, gold_data):
        """计算多任务组合奖励"""
        try:
            # 任务1: 总分准确性
            total_reward = self._evaluate_task(
                self._build_total_score_prompt(prompt, generated_text, gold_data)
            )
            # 任务2: 步骤分合理性
            step_reward = self._evaluate_task(
                self._build_step_score_prompt(prompt, generated_text, gold_data)
            )
            # 任务3: 错误识别
            error_reward = self._evaluate_task(
                self._build_error_prompt(prompt, generated_text, gold_data)
            )
            
            # 加权组合
            combined_reward = 0.4 * total_reward + 0.4 * step_reward + 0.2 * error_reward
            logger.debug(f"奖励分解 - 总分:{total_reward:.3f}, 步骤:{step_reward:.3f}, 错误:{error_reward:.3f}, 综合:{combined_reward:.3f}")
            return combined_reward
            
        except Exception as e:
            logger.warning(f"奖励计算失败: {e}")
            return 0.3  # 默认奖励
    
    def _evaluate_task(self, task_prompt):
        """评估单个任务"""
        try:
            inputs = self.tokenizer(
                task_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['input_ids'], 
                    max_new_tokens=5,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
                
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            reward = self._compute_reward_from_generated_text(generated_text, task_prompt)
            return reward
        except Exception as e:
            logger.warning(f"任务评估失败: {e}")
            return 0.5
    
    def _compute_reward_from_generated_text(self, generated_text, original_prompt):
        """改进的奖励计算逻辑"""
        # 提取模型的实际回答
        if generated_text.startswith(original_prompt):
            answer = generated_text[len(original_prompt):].strip()
        else:
            answer = generated_text
    
        # 更严格的判断逻辑
        answer_lower = answer.lower().strip()
    
        # 明确匹配"是"或"否"
        if answer_lower in ["是", "是的", "对", "正确", "准确", "合理"]:
            return 1.0
        elif answer_lower in ["否", "不是", "不对", "错误", "不准确", "不合理"]:
            return 0.0
        elif "是" in answer_lower and "不" not in answer_lower and "否" not in answer_lower:
            return 1.0
        elif "否" in answer_lower or "不" in answer_lower:
            return 0.0
        else:
            # 无法判断的情况
            print(f"无法判断的回答: '{answer}'")
            return 0.3  # 给较低的默认分
    
    def _build_total_score_prompt(self, prompt, generated_text, gold_data):
        pred_total = self._extract_json_value(generated_text, 'manual_label')
        return f"""评估数学评分模型的总分预测准确性:

题目总分: {gold_data['total']}
人工标注总分: {gold_data['manual_label']}
模型预测总分: {pred_total}

模型预测的总分是否准确？
请只回答一个字"是"或"否"
回答：
"""
    
    def _build_step_score_prompt(self, prompt, generated_text, gold_data):
        true_scores = [step['label'] for step in gold_data['steps'][:3]]
        pred_scores = self._extract_step_scores(generated_text)[:3]
        return f"""评估数学评分模型的步骤分合理性:

人工步骤分: {true_scores}
模型步骤分: {pred_scores}

模型步骤评分是否合理？
请只回答一个字"是"或"否"
回答："""
    
    def _build_error_prompt(self, prompt, generated_text, gold_data):
        true_errors = []
        for step in gold_data['steps'][:3]:
            true_errors.extend(step['errors'])
        pred_errors = self._extract_errors(generated_text)[:5]
        return f"""评估数学评分模型的错误识别:

人工标注错误: {true_errors}
模型识别错误: {pred_errors}

模型错误识别是否准确？
请只回答一个字"是"或"否"
回答：
"""
    
    def _extract_json_value(self, text, key):
        try:
            json_str = find_valid_json(text)
            if json_str:
                data = json.loads(json_str)
                value = data.get(key, '未找到')
                return str(value) if value is not None else '未找到'
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}")
        return '解析失败'
    
    def _extract_step_scores(self, text):
        try:
            json_str = find_valid_json(text)
            if json_str:
                data = json.loads(json_str)
                steps = data.get('steps', [])
                scores = []
                for step in steps[:5]:
                    score = step.get('step_score', '未知')
                    scores.append(str(score) if score is not None else '未知')
                return scores
        except Exception as e:
            logger.warning(f"步骤分解析失败: {e}")
        return ['解析失败']
    
    def _extract_errors(self, text):
        try:
            json_str = find_valid_json(text)
            if json_str:
                data = json.loads(json_str)
                steps = data.get('steps', [])
                errors = []
                for step in steps[:5]:
                    step_errors = step.get('errors', [])
                    if isinstance(step_errors, list):
                        errors.extend([str(e) for e in step_errors])
                return errors
        except Exception as e:
            logger.warning(f"错误解析失败: {e}")
        return ['解析失败']


# ===========================================================
# ✅ 构建评分 Prompt
# ===========================================================
import json

def build_scoring_prompt(item):
    # Few-shot 示例
    few_shot_examples = [
        {
            "question": "计算等差数列的前n项和，已知首项a₁=2，公差d=3，项数n=5",
            "reference": "使用公式 Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40",
            "analysis": "正确应用等差数列求和公式，计算准确",
            "total": 10,
            "student_answer": [
                {"response": "Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40"}
            ],
            "output": {
                "total": 10,
                "steps": [
                    {
                        "response": "Sₙ = n/2 × [2a₁ + (n-1)d] = 5/2 × [4 + 12] = 40", 
                        "step_score": 2, 
                        "errors": ["步骤正确"]
                    }
                ],
                "manual_label": 10
            }
        }
    ]

    # 构建few-shot示例块
    few_shot_meta = {
        "prefix": "【参考示例】",
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


def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    prompts, gold_data_list = [], []
    for item in tqdm(data, desc="加载数据"):
        prompt = build_scoring_prompt(item)
        prompts.append(prompt)
        gold_data_list.append(item)
    return prompts, gold_data_list


def create_grpo_dataset(prompts, tokenizer, max_length=2048):
    processed_data = []
    for prompt in tqdm(prompts, desc="处理数据"):
        encoding = tokenizer(prompt, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        processed_data.append({
            "prompt": prompt,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
        })
    return Dataset.from_list(processed_data)


# ===========================================================
# ✅ 修复的GRPO奖励函数 - 使用函数而不是类
# ===========================================================
# 全局变量用于存储奖励模型和数据映射
_reward_model = None
_prompt_to_gold = None

def create_reward_function(reward_model_path, prompts_list, gold_data_list):
    """创建奖励函数 - 返回一个普通函数"""
    global _reward_model, _prompt_to_gold
    
    # 初始化奖励模型
    _reward_model = MultiTaskRewardModel(reward_model_path)
    _prompt_to_gold = {p: g for p, g in zip(prompts_list, gold_data_list)}
    
    logger.info(f"奖励函数初始化完成，共{len(_prompt_to_gold)}个样本")
    
    # 返回实际的奖励函数
    def math_reward_func(prompts, completions, **kwargs):
        """GRPO 奖励函数：根据 prompt 与模型生成的 completion 计算奖励"""
        rewards = []

        logger.info(f"计算奖励，共{len(prompts)}个样本")

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            if i % 10 == 0:
                logger.info(f"奖励计算进度: {i}/{len(prompts)}")

            try:
                # 找出对应的 gold_data
                gold_data = _prompt_to_gold.get(prompt, None)

                if gold_data is None:
                    rewards.append(0.3)  # 默认奖励
                    continue

                # 调用奖励模型计算奖励
                reward = _reward_model.compute_reward(prompt, completion, gold_data)
                rewards.append(reward)

            except Exception as e:
                logger.warning(f"单个样本奖励计算失败: {e}")
                rewards.append(0.3)

        # 返回torch.Tensor
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        print(reward_tensor)
        logger.info(f"奖励计算完成，平均奖励: {reward_tensor.mean().item():.3f}")
        
        return reward_tensor
    
    return math_reward_func


# ===========================================================
# ✅ 主训练流程
# ===========================================================
def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 清理GPU内存
    torch.cuda.empty_cache()
    model_name = r"/root/autodl-tmp/LLaMA-Factory/lora_to_qwen/loraed_qwen3"
    reward_model_path = r"/root/autodl-tmp/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    dataset_path = r"/root/autodl-tmp/GRPO/lora_train/data/7_Math_ShortAns.jsonl"

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )

    # 加载数据
    prompts_list, gold_data_list = load_dataset(dataset_path)
    
    # 创建训练数据
    training_data = create_grpo_dataset(prompts_list, tokenizer)
    
    # 创建奖励函数（返回普通函数）
    reward_function = create_reward_function(reward_model_path, prompts_list, gold_data_list)

    # GRPO配置
    grpo_config = GRPOConfig(
       # 数据预处理
    remove_unused_columns=False,
    max_prompt_length=1024,
    num_generations=2,
    max_completion_length=512,
    
    # 训练参数
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    
    # GRPO核心参数
    beta=0.15,
    epsilon=0.15,
    
    # 奖励设置
    scale_rewards=True,
    
    # 生成设置
    temperature=0.7,
    top_p=0.9,
    
    # 输出和日志
    output_dir=os.path.join(r"/root/autodl-tmp/GRPO/lora_train/output", "qwen3_math_grpo_output4"),
    logging_strategy="steps",
    logging_dir=os.path.join(r"/root/autodl-tmp/GRPO/lora_train/output", "GRPO_logs4"),
    logging_steps=50,

    save_steps=50,
    report_to="tensorboard",
    gradient_checkpointing=True,
    
    # 其他
    shuffle_dataset=True,
)
    

    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,  # 现在传递的是函数，不是类实例
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=training_data,
    )

    logger.info("开始GRPO训练...")
    trainer.train(resume_from_checkpoint="/root/autodl-tmp/GRPO/lora_train/output/qwen3_math_grpo_output4/checkpoint-200")

    final_dir = os.path.join(r"/root/autodl-tmp/GRPO/lora_train/output", "qwen3_math_grpo_model4")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"训练完成！模型保存在: {final_dir}")


if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
from data_process import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
本模型是想将数学题分解为多个子任务，每个子任务都有一个奖励模型，最后将所有子任务的奖励模型的奖励相加作为最终奖励。
用于判断大任务分解成小任务是否可以增强奖励模型总体效果
'''

class MultiTaskRewardModel:
    """多任务奖励模型 - 基于大语言模型的判断"""
    
    def __init__(self, model_path):
        '''
        model_path是奖励模型的路径
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if  self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()#参数不需要更新

    def compute_reward(self,prompt,generated_text,gold_data):
        
        try:
            #子任务1：判断总分是否正确
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
            print(f"奖励模型计算奖励时出错: {e}")
            return 0.3
    
    def _evaluete_task(self,task_prompt):
        
        try:
            inputs = self.tokenizer(
                task_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=5,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            score = self._compute_reward_from_generated_text(generated_text,task_prompt)
            return score
        except Exception as e:
            print(f"奖励模型计算奖励时出错: {e}")
            return 0.3
        
    def _compute_reward_from_generated_text(self,generated_text,task_prompt):
        #提取模型的实际回答
        if generated_text.startswith(task_prompt):
            answer = generated_text[len(task_prompt):]
        else:
            answer = generated_text

        answer = answer.strip()

        # 明确匹配"是"或"否"
        if answer in ["是", "是的", "对", "正确", "准确", "合理"]:
            return 1.0
        elif answer in ["否", "不是", "不对", "错误", "不准确", "不合理"]:
            return 0.0
        elif "是" in answer and "不" not in answer and "否" not in answer:
            return 1.0
        elif "否" in answer or "不" in answer:
            return 0.0
        else:
            # 无法判断的情况
            print(f"无法判断的回答: '{answer}'")
            return 0.3  # 给较低的默认分
        
    #子任务1prompt的构建
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
        
        
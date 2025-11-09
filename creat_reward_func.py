import torch
from Mutiwork_RM import *
from data_process import *

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
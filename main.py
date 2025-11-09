from creat_reward_func import *
from Mutiwork_RM import *
from data_process import *
from trl import GRPOConfig, GRPOTrainer

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 清理GPU内存
    torch.cuda.empty_cache()
    model_name = r"your model path"
    reward_model_path = r"your reward model path"
    dataset_path = r"your dataset path"#在本示例中是7_Math_ShortAns.jsonl

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
    training_data = create_train_dataset(prompts_list, tokenizer)
    
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
    trainer.train()

    final_dir = os.path.join(r"/root/autodl-tmp/GRPO/lora_train/output", "qwen3_math_grpo_model4")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"训练完成！模型保存在: {final_dir}")


if __name__ == "__main__":
    main()
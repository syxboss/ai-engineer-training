The Loop Explained:

Algorithm to Agent (via Trainer): The Algorithm (the "brain") creates an improved Prompt Template and selects Tasks. The Trainer then sends both to the Trainer.

Agent to Algorithm (via Trainer): For each task it receives, the Agent uses the provided prompt template to perform a Rollout, executing its logic and potentially using tools. During this rollout, the runner that runs the agent captures Spans that detail every step. The agent also calculates a Reward for its performance on the task. These spans and rewards are then sent back to the Algorithm via the Trainer.

Algorithm Learning: The Algorithm then analyzes these spans and rewards to learn how to improve the agent's behavior, for example, by generating a better prompt. This improved prompt is then used in the next iteration of tasks.

Algorithm  -> Agent: 生成新的 提示词模板，并挑选任务， 给 Trainer
Agent  -> Trainer: Agent 使用上面的提示词 执行 Rollout ，执行之后 Runner 捕获所有的 Spans 然后计算出新的 Reward。 随后吧 Span 和 Reward 发送给 Algorithm。
Algorithm 学习过程: 基于 Spans 和 Reward，算法生成新的提示词模板 进入下一轮循环



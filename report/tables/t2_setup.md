| setting | value |
|---|---|
| start | supplied PPO midpoint policy LoRA and value model |
| value model | Qwen/Qwen2.5-0.5B-Instruct with LoRA + scalar head, trainable weights in fp32 |
| updates | standard 20, forks 8, 1 prompt per update, 2 PPO epochs |
| learning rates | policy 3e-06, value LoRA 0.0001, value head 0.0003 |
| clipping | epsilon 0.2; study [0.05, 0.2, 0.5] |
| KL penalty | beta_KL 0.1; study [0.0, 0.1, 0.2] |
| GAE | gamma 1.0, lambda 0.95; value coef 0.5 |
| missing EOS penalty | 1.0 |
| lengths | prompt 256, train response 512, eval response 768, reward input 1280 |
| grad clip | 1.0 |
| prompts | 1200 train pool (seeded shuffle, update k uses k-th prompt), 200 eval |
| cached batch | 32 supplied rollouts, 8814 response tokens |

| setting | value |
|---|---|
| policies | SFT (base, no adapter), standard DPO, standard PPO, standard GRPO |
| prompts | XSTest, 450 (250 safe, 200 unsafe), 10 safe and 8 unsafe categories |
| decoding | greedy, 256 new tokens |
| AI judge | Qwen/Qwen2.5-3B-Instruct, greedy, 64 tokens, 5 labels |
| manual audit | 240 responses (60 prompts x 4 policies; 30 safe, 30 unsafe, seed 6304), manual human reviewed blind to AI label and prompt class |

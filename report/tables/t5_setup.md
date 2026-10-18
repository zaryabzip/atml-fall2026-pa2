| setting | value |
|---|---|
| policies | SFT (base), RLVR and RLAIF (supplied adapters) |
| data | GSM8K eval 300, SVAMP transfer 100, controlled diagnostics 100 (20 problems x 5 variants) |
| decoding | greedy, 512 new tokens |
| verifier | exact match of the number after the last #### marker |
| AI judge | Qwen/Qwen2.5-3B-Instruct, pairwise, greedy, 4 tokens, A/B order set by hash |
| win rate | win 1, tie 0.5, loss 0, each policy vs SFT on the same problem |

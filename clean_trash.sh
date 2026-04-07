BASE="/data/code/exp/EvolAnalsis"                                                                                                                                                       
                                                                                                                                                                                        
# -------------------------------------------------------                                                                                                                               
# 1. Bug-affected 0409_cap 模型权重 (36G)                                                                                                                                               
#    (保留 Trajectory/ 目录用于对比分析)                                                                                                                                                
# -------------------------------------------------------                                                                                                                               
rm -rf "$BASE/checkpoints/agentevolver/webshop_3b_duet_0409_cap/global_step_100"                                                                                                        
                                                                                                                                                                                        
# -------------------------------------------------------                                                                                                                               
# 2. 0410 失败的运行 — 只跑了10步，磁盘满崩溃 (~0.5G)                                                                                                                                   
# -------------------------------------------------------                                                                                                                               
rm -rf "$BASE/checkpoints/agentevolver/webshop_3b_duet_0410_cap"                                                                                                                        
rm -rf "$BASE/checkpoints/agentevolver/webshop_3b_duet_0410_bell"                                                                                                                       
rm -rf "$BASE/checkpoints/agentevolver/webshop_3b_duet_0410_ema_cap"                                                                                                                    
rm -rf "$BASE/experiments/webshop_3b_duet_0410_cap"                                                                                                                                     
rm -rf "$BASE/experiments/webshop_3b_duet_0410_bell"                                                                                                                                    
rm -rf "$BASE/experiments/webshop_3b_duet_0410_ema_cap"                                                                                                                                 
                                                                                                                                                                                        
# -------------------------------------------------------                                                                                                                               
# 3. 老 SciWorld 预实验 (~201G)                                                                                                                                                         
#    全部是 paper config 之前的探索性跑                                                                                                                                                 
# -------------------------------------------------------                                                                                                                               
rm -rf "$BASE/experiments/sciworld_3b_grpo_dr3_v3aug_teacher72b_bz8_ntr1_gap_gate_2400"                                                                                                 
rm -rf "$BASE/experiments/sciworld_7b_grpo_bz8"                                                                                                                                         
rm -rf "$BASE/experiments/sciworld_3b_grpo_bz8_2400"                                                                                                                                    
rm -rf "$BASE/experiments/sciworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random_binary_reward"                                                                                    
rm -rf "$BASE/experiments/sciworld_3b_grpo_chord_bz8_mix1_mu_decay_50_wo_kl_2400"                                                                                                       
rm -rf "$BASE/experiments/sciworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random"                                                                                                  
rm -rf "$BASE/experiments/sciworld_7b_grpo_bz8_standard70"                                                                                                                              
rm -rf "$BASE/experiments/sciworld_3b_grpo_bz8"                                                                                                                                         
rm -rf "$BASE/experiments/sciworld_7b_grpo_bz8_standard50"                                                                                                                              
rm -rf "$BASE/experiments/sciworld_3b_grpo_luffy_bz8_mix1_800"                                                                                                                          
rm -rf "$BASE/experiments/sciworld_3b_grpo_dr3_v3aug_teacher72b_bz8_ntr1_gap_gate"                                                                                                      
rm -rf "$BASE/experiments/sciworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__grpo_teacher_baseline_sep_v1"                                                                    
rm -rf "$BASE/experiments/sciworld_3b_grpo_dr3_v3aug_teacher72b_bz8_ntr1_gap_gate_800_new"                                                                                              
rm -rf "$BASE/experiments/sciworld_3b_grpo_dr3_v3aug_teacher72b_bz8_ntr1_gap_gate_800"                                                                                                  
rm -rf "$BASE/experiments/sciworld_3b_grpo_bz8_800"  


# -------------------------------------------------------                                                                                                                               
# 4. 老 ALFWorld 预实验 logs (~95G)                                                                                                                                                     
#    gate/analysis/exp_replay/7b 等探索性跑                                                                                                                                             
# -------------------------------------------------------                                                                                                                               
rm -rf "$BASE/experiments/alfworld_3b_grpo_teacher72b_only_bz8_mix1"*                                                                                                                   
rm -rf "$BASE/experiments/alfworld_3b_grpo_exp_replay"*                                    
rm -rf "$BASE/experiments/alfworld_3b_grpo_bz16_stricter"                                  
rm -rf "$BASE/experiments/alfworld_3b_grpo_dr3_teacher72b_bz8_ntr1"                        
rm -rf "$BASE/experiments/alfworld_7b_grpo_baseline"                                       
rm -rf "$BASE/experiments/alfworld_3b_grpo_bz8"                                            
rm -rf "$BASE/experiments/webshop_7b_grpo_bz8"                                             

# -------------------------------------------------------
# 5. 老 ALFWorld 预实验 checkpoints (~3G, 无模型权重)                                      
# -------------------------------------------------------                                  
rm -rf "$BASE/checkpoints/agentevolver/alfworld_3b_grpo_teacher72b_only"*
rm -rf "$BASE/checkpoints/agentevolver/alfworld_3b_grpo_exp_replay"*                       
                                                                 
# -------------------------------------------------------                                  
# 6. 旧 wandb 本地缓存 — 2026年1-2月 (~25G)                      
#    数据已同步到 wandb 云端                
# -------------------------------------------------------                                  
rm -rf "$BASE/wandb/run-202601"*            
rm -rf "$BASE/wandb/run-202602"*            
                                                                 
# -------------------------------------------------------                                  
# 验证                                                           
# -------------------------------------------------------                                  
echo ""
echo "清理完成，当前磁盘使用:"              
df -h /data                



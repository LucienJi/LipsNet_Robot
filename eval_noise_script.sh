# model_types=("LipsNet" "BCMLP" "Expert" "RMA")
model_types=("LipsNet" "LipsNet" "BCMLP" "Expert" "RMA")
# eval_names=("LipsNet" "BCMLP" "Expert" "RMA")
eval_names=("LipsNet" "L1Penalty" "BCMLP" "Expert" "RMA")
model_paths=("logs/LipsNet/Mar01_14-37-49_wReward/model_5000.pt" \
            "logs/LipsNet/Mar04_09-11-08_L1Penalty/model_5000.pt"\
            "logs/BCMLP/Mar01_14-47-42_wReward/model_5000.pt" \
            "logs/Expert/Feb29_00-18-55_wReward/model_5000.pt" \
            "logs/RMA/Mar01_15-04-55_wReward/model_5000.pt")


# model_types=("LipsNet" "BCMLP")
# eval_names=("LipsNet" "BCMLP")
# model_paths=("logs/LipsNet/Mar01_14-37-49_wReward/model_5000.pt" \
#             "logs/BCMLP/Mar01_14-47-42_wReward/model_5000.pt" )


cmd_vel_values=(0.5 1.0)
save_paths=("logs/Eval/Flat/Eval_noise_percent/noiseV05" "logs/Eval/Flat/Eval_noise_percent/noiseV10")

num_task=${#cmd_vel_values[@]}
num_models=${#model_types[@]}
# 启动任务计数器
task_counter=0

# noise_levels=(0 1 2 3 4 5 6 7 8 9 10)
noise_levels=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
noise_types=("gaussian" "uniform")



for ((i=0; i<$num_task; i++))
do
    cmd_vel=${cmd_vel_values[$i]}
    save_path=${save_paths[$i]}

    # for baseline_name in "${baseline_names[@]}"
    for ((j=0; j<$num_models; j++))
    do
        model_type=${model_types[$j]}
        model_path=${model_paths[$j]}
        eval_name=${eval_names[$j]}
        for ((k=0; k<${#noise_levels[@]}; k++))
        do 
            noise_level=${noise_levels[$k]}
            for ((l=0; l<${#noise_types[@]}; l++))
            do 
                noise_type=${noise_types[$l]}
                # 计算 CUDA 设备 ID
                device_id=$((task_counter % 2 + 1))
                rl_device="cuda:${device_id}"
                sim_device="cuda:${device_id}"
                # 使用日期和时间为输出文件生成唯一的前缀
                timestamp=$(date +"%Y%m%d%H%M%S")
                output_file="output_${eval_name}_${noise_type}_${noise_level}_${timestamp}.out"
                nohup python eval_noise.py \
                    --headless \
                    --rl_device "cuda:${device_id}" \
                    --sim_device "cuda:${device_id}" \
                    --model_type "$model_type" \
                    --model_path "$model_path" \
                    --cmd_vel "$cmd_vel" \
                    --eval_name "$eval_name" \
                    --eval_path "$save_path" \
                    --noise_level "$noise_level" \
                    --noise_type "$noise_type" \
                    --flat_terrain # store true  
                    > "$output_file" 2>&1 &
                    # 更新任务计数器
                    task_counter=$((task_counter + 1))
                    # 可以选择在此处检查当前运行的后台任务数量
                    # 并在达到一定数量时等待它们完成
                    if (( task_counter % 8 == 0 )); then
                        wait
                    fi
            done
        done
    done
done

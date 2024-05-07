model_types=("LipsNet" "LipsNet")
eval_names=("LipsNet_woReward" "LipsNet_wReward")
model_paths=("logs/LipsNet/Dec20_13-57-35_woReward/model_10000.pt" \
            "logs/LipsNet/Dec20_13-56-05_wReward/model_10000.pt")


cmd_vel_values=(0.5 1.0)
save_paths=("logs/Eval/noiseV05" "logs/Eval/noiseV10")

num_task=${#cmd_vel_values[@]}
num_models=${#model_types[@]}
# 启动任务计数器
task_counter=0

noise_level="--noise_level 1 2 3 4 5 6 7 8 9 10"




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
        # 计算 CUDA 设备 ID
        device_id=$((task_counter % 2 + 1))
        rl_device="cuda:${device_id}"
        sim_device="cuda:${device_id}"
        # 使用日期和时间为输出文件生成唯一的前缀
        timestamp=$(date +"%Y%m%d%H%M%S")
        output_file="output_${eval_name}_${timestamp}.out"
        nohup python eval_noise.py \
            --headless \
            --rl_device "cuda:${device_id}" \
            --sim_device "cuda:${device_id}" \
            --model_type "$model_type" \
            --model_path "$model_path" \
            --cmd_vel "$cmd_vel" \
            --eval_name "$eval_name" \
            --eval_path "$save_path" \
            $noise_level \
            > "$output_file" 2>&1 &
            # 更新任务计数器
            task_counter=$((task_counter + 1))
            # 可以选择在此处检查当前运行的后台任务数量
            # 并在达到一定数量时等待它们完成
            if (( task_counter % 6 == 0 )); then
                wait
            fi
    done
done

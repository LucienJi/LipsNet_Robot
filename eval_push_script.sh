model_types=("LipsNet" "BCMLP" "Expert" "RMA")
eval_names=("LipsNet" "BCMLP" "Expert" "RMA")
model_paths=("logs/LipsNet_v2/Dec21_16-41-55_wReward/model_20000.pt" \
            "logs/BCMLP_v2/Dec21_16-41-07_wReward/model_20000.pt" \
            "logs/Expert/Dec19_15-24-37_wReward/model_10000.pt" \
            "logs/RMA/Dec19_15-28-51_wReward/model_10000.pt")

num_modeles=${#model_types[@]}

cmd_vel_values=(0.5 1.0)
save_paths=("logs/Eval_push_duration/pushV05" "logs/Eval_push_duration/pushV10")
num_vel=${#cmd_vel_values[@]}


push_forces=(10 15 20 25 30 35 40 45 50 55 60)
push_intervals=(100)
push_durations=(25 50)

# 启动任务计数器
task_counter=0
for ((j=0; j<$num_modeles; j++))
do
    model_type=${model_types[$j]}
    eval_name=${eval_names[$j]}
    model_path=${model_paths[$j]}

    for ((i=0; i<$num_vel; i++))
    do  
        cmd_vel=${cmd_vel_values[$i]}
        save_path=${save_paths[$i]}
        for push_force in "${push_forces[@]}"
        do
            for push_duration in "${push_durations[@]}"
            do  
                device_id=$((task_counter % 2 + 1))
                rl_device="cuda:${device_id}"
                sim_device="cuda:${device_id}"
                # 使用日期和时间为输出文件生成唯一的前缀
                timestamp=$(date +"%Y%m%d%H%M%S")
                output_file="output_${eval_name}_${push_force}_${push_duration}${timestamp}.out"
                nohup python eval_push.py --headless\
                                        --rl_device "cuda:${device_id}" \
                                        --sim_device "cuda:${device_id}" \
                                        --model_type $model_type \
                                        --model_path $model_path \
                                        --cmd_vel $cmd_vel \
                                        --push_force $push_force \
                                        --push_duration $push_duration\
                                        --eval_path "$save_path" \
                                        --eval_name "$eval_name" \
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

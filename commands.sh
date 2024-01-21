








# clip  UCF

nohup python3 main.py --cfg config/clipclap.yaml \
                        --device cuda:6 \
                        --root_dir /home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/UCF  \
                        --log_dir /shared-network/aoq234/hip_gzsl_experiments/ClipClap_UCF \
                        --dataset_name UCF \
                        --epochs 20 \
                        --lr 0.00007 \
                        --use_wavcaps_embeddings True \
                        --modality both  \
                        --word_embeddings both   \
                        --run all > logs/ClipClap_UCF.log &




python3 get_evaluation.py --cfg config/clipclap.yaml     \
                --load_path_stage_A /shared-network/aoq234/hip_gzsl_experiments/ClipClap_UCF_ablation_w_clip/Oct23_08-07-58_952980_callisto     \
                --load_path_stage_B /shared-network/aoq234/hip_gzsl_experiments/ClipClap_UCF_ablation_w_clip/Oct23_08-59-42_055708_callisto     \
                --dataset_name UCF       \
                --root_dir /home/aoq234/akata-shared/aoq234/avzsl/avgzsl_benchmark_datasets/UCF       \
                --device cuda:5















########## ActivityNet #################


python3 evaluate_clip.py --cfg config/clipclap.yaml \
                            --load_path_stage_B logs/clip/activitynet/orinal_clip  \
                            --eval_save_performances False \
                            --dataset_name ActivityNet \
                            --root_dir /home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/ActivityNet \
                            --use_wavcaps_embeddings True \
                            --eval_modality both \
                            --device cuda:2


nohup python3 main.py --cfg config/clipclap.yaml \
                        --device cuda:6 \
                        --root_dir /home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/ActivityNet  \
                        --log_dir /shared-network/aoq234/hip_gzsl_experiments/ClipClap_ActivityNet \
                        --dataset_name ActivityNet \
                        --epochs 15 \
                        --lr 0.0001 \
                        --use_wavcaps_embeddings True \
                        --modality both  \
                        --word_embeddings both   \
                        --run all > logs/ClipClap_ActivityNet.log &








# VGGSound


python3 evaluate_clip.py --cfg config/clipclap.yaml \
                            --load_path_stage_B logs/clip/vggsound/original_clip  \
                            --eval_save_performances False \
                            --dataset_name VGGSound \
                            --root_dir /home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/VGGSound \
                            --use_wavcaps_embeddings True \
                            --eval_modality audio \
                            --device cuda:2


nohup python3 main.py --cfg config/clipclap.yaml \
                        --device cuda:5 \
                        --root_dir /home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/VGGSound  \
                        --log_dir /shared-network/aoq234/hip_gzsl_experiments/ClipClap_VGGSound \
                        --dataset_name VGGSound \
                        --epochs 15 \
                        --lr 0.0001 \
                        --use_wavcaps_embeddings True \
                        --modality both  \
                        --word_embeddings both   \
                        --run all > logs/ClipClap_VGGSound.log &

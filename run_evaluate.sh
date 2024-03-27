export PYTHONPATH=pose_extractor/build/:$PYTHONPATH

data="train_test_data_balance_"

for index in $(seq 1 10)
do
    # python3 ./Subroutine/split_data.py --folder $data$index #產生資料集
    # python3 ./Subroutine/export_data_train.py --folder $data$index #提取訓練集的特徵
    # python3 ./Subroutine/export_data_test.py --folder $data$index #把測試集的影像轉成特徵

    # 模型辨識
    python3 ./Evaluate/Rule.py --folder $data$index
    python3 ./Evaluate/LR.py --folder $data$index
    python3 ./Evaluate/RF.py --folder $data$index
    python3 ./Evaluate/NB.py --folder $data$index
    python3 ./Evaluate/NN.py --folder $data$index
    python3 ./Evaluate/KNN.py --folder $data$index
    python3 ./Evaluate/DT.py --folder $data$index
done

### Part 1 ###

=> 進入 Make_data 資料夾
cd ./Make_data

=> 影片標記與分割
python3 cut_by_mark.py

=> 整合
python3 mix_all_data.py

=> 平衡資料集
python3 balance_normal_data.py

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### Part 2 ###

=> 回到根目錄資料夾
cd ../

=> 資料分割
python3 ./Subroutine/split_data.py --folder xxx

=> 設定環境變數
export PYTHONPATH=pose_extractor/build/:$PYTHONPATH

=> 提取訓練集的特徵
python3 ./Subroutine/export_data_train.py --folder xxx

=> 把測試集的影像轉成特徵
python3 ./Subroutine/export_data_test.py --folder xxx

備註：--> xxx是產生的資料夾名稱,單次流程必須相同

(需要的時候在執行)畫盒鬚圖的數據 => 提取原始的特徵數值
python3 ./Subroutine/export_data_train_original_value.py --folder train_test_data_balance_1

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### Part 3 ###

=> 模型辨識
python3 ./Evaluate/Rule.py --folder xxx
python3 ./Evaluate/LR.py --folder xxx
python3 ./Evaluate/RF.py --folder xxx
python3 ./Evaluate/NB.py --folder xxx
python3 ./Evaluate/NN.py --folder xxx
python3 ./Evaluate/KNN.py --folder xxx
python3 ./Evaluate/DT.py --folder xxx

=> 整體的辨識的結果會在 ./Result/xxx/evaluate.csv
=> 詳細的辨識結果在 ./Result/xxx/single

備註：--> xxx是產生的資料夾名稱,單次流程必須相同

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### Part 4 ###

=> Part 2 、 Part 3可以使用 run_evaluate.sh 批次執行
sh run_evaluate.sh

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### Part 5 ###

=> 測試
sudo PYTHONPATH=pose_extractor/build/:$PYTHONPATH python3 main_alert.py

=> 偵測結果會在 ./Result/alert
# WriteId
书写者识别任务

2019.5.23
1. 完成数据的处理工作，每个训练样本为100个RHS
2. 完成Data10基本框架，测试集精度达83%，训练集达92%：
  - 5层LSTM堆叠
  - batch_size=1200, Adam lr=0.001, epochs=60
3. TODO:
  - 精度提升，参数初始化等
 
2019.5.24
1. 完成测试脚本，问题：单个 100RHS 测试准确率低，TODO：多个RHS共同判断

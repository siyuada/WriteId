# WriteId
书写者识别任务

2019.5.23
1. 完成数据的处理工作，每个训练样本为100个RHS
2. 完成Data10基本框架，测试集精度达83%，训练集达92%：
  - 6层LSTM堆叠
  - batch_size=800, Adam lr=0.001, epochs=60
3. TODO:
  - 精度提升，参数初始化等
 
2019.5.24
1. 完成测试脚本，问题：单个 100RHS 测试准确率低（由于分类顺序的问题，已解决，单个80%左右），TODO：多个RHS共同判断 5-10个RHS可以达到100%精度
2. 测试不同层数
  - 6层，batch_size=1000, 70epochs，训练集精度：97.87%，测试集85.85%，模型rnn6,Dropout=0.5
  - 8层，batch_size=800，基本学不到东西，可能由于层数太多，梯度消失
  
2019.5.27
1. 100分类任务：
  双向LSTM，3层，隐藏层400(x2)，精度可以得到有效提升
  batchsize=2000，Adam加入权重衰减0.0005

2019.5.28
1. 数据清洗：
  在每个人生成RHS时，若样本中笔画数仅为1且笔画位置仅为1，即该汉字只录入了一个点，舍弃。已完成。
2. 可以考虑对数据文件生成索引文件，否则每次数据读取耗时较多（但也无所谓...）
3. 重新对清新后数据和调整数据后数据进行训练，记录参数
  - Task10: 60epochs
    - 每个人样本的序列长度：100，模型保存名rnn5.pkl(被误删了TAT)，曲线图：10-5layer-1000ba.jpg
    - 5层单向，隐藏层=800，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0.0005（在后期降低学习率）
    - 准确率：训练：89.01%，测试 81.30%， Loss：训练：0.2887，测试：0.4891
    
    70epochs
    - 每个人样本的序列长度：100，模型保存名rnn6.pkl，曲线图：10-6layer-1000ba.jpg
    - 6层单向，隐藏层=800，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0.0005（在后期降低学习率）
    - 准确率：训练：90.57%，测试 82.93%， Loss：训练：0.2574，测试：0.4415
    
    测试效果：
    - 2018310898与2018310874是比较容易分错的，是否可以对错误样本做可视化分析？？？#TODO，其他类别还行，取10个基本能保证90%的正确率。
    
    对于2018310898的测试样本的判断结果: 应该为8，而5占大多数，判断错误
    ```
    tensor([5, 5, 8, 5, 8, 8, 5, 8, 5, 5, 8, 8, 5, 5, 5, 5, 8, 2, 5, 5, 8, 2, 8, 5,
        5, 8, 5, 8, 5, 5, 8, 5, 8, 5, 8, 8, 8, 8, 5, 8, 8, 5, 5, 8, 5, 8, 8, 2,
        8, 5, 5, 5, 5, 8, 5, 8, 8, 5, 5, 8, 8, 5, 8, 5, 5, 5, 8, 5, 2, 8, 8, 5,
        5, 5, 5, 8, 8, 5, 8, 2, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        8, 5, 8, 8], device='cuda:0')
    5
    ```
    
    对于2018310874的测试样本的判断结果:5依然占大多数，判断正确
    ```
    tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5, 5, 5,
        5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 8,
        5, 5, 5, 8, 5, 8, 5, 6, 5, 8, 5, 5, 5, 5, 5, 8, 5, 5, 8, 5, 5, 8, 2, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 8], device='cuda:0')
    5
    ```
    60epochs
    - 每个人样本的序列长度：100，模型保存名10rnn1-bi.pkl，曲线图：10-1layer-bi.jpg
    - 1层双向，隐藏层=400，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0.0005（在后期降低学习率）
    - 准确率：训练：95.12%，测试 84.21%， Loss：训练：0.1463，测试：0.4825
    - 过拟合严重，加入防止过拟合
    测试效果：
    与单向6层相同，判断8样本时容易判断为5！
    
    对1层双向加0.5的Dropout，60epochs
    - 准确率：训练：95.49%，测试84.73%， Loss：训练：0.1287，测试：0.50
    提升不大？？
    - 考虑对类别5和8增加样本？
    
    对5和8两个类别增加样本，训练样本数为1600
     - 双向2层训练，50轮,0.5的Dropout
     - 准确率：训练：93.04%，测试：86.30%， Loss：训练：0.1915，测试：0.3829
     - 感觉还能继续下降，在基础上继续训练看看
     - 准确率：训练：96.85%，测试：87.00%， Loss：训练：0.0915，测试：0.446
     - **感觉应该要用其他的特征避免5,8混淆，但在测试阶段取10个RHS基本能保证分对，仍有错分现象。**
    
  - Task100：60epochs
    - 每个人样本的序列长度：100，模型保存名100rnn3.pkl，曲线图：100-1.jpg
    - 3层双向，隐藏层=400，batch_size=1000，训练样本数1500，测试600，Adam学习率0.001，权重衰减0.0005（在后期降低学习率）
    - 准确率：训练：95.98%，测试：91.70%， Loss：训练：0.1213，测试：0.2532
    
    测试效果：取十个样本取多数，93%正确率，即判断有7个人错误
    错误分类
     ```
    (array([100, 101, 102, 103, 104, 105, 106]),)
    [2018311127 2018311146 2018312459 2018312470 2018312476 2018312481 2018312484]
    [2018211051 2018211080 2018210461 2018310898 2018310907 2018210461 2018310927]
    ```
    TAT发现是因为一共有107个类别，只做了100个人的分类，没看清题目和数据的后果...再训一次咯，证明前100个人均分类正确了！！
    - 107重新训练，上述参数不变
    - Train Loss: 0.142875, Train Acc: 0.952950, Eval Loss: 0.311687, Eval Acc: 0.900477
    
    测试效果：测试了5次，均为100%。
    
    
 2019.5.28-Rui
 
1. 重新排序并生成Task10文件夹，取1间隔RHS生成Task10_2
   
2. 双向LSTM模型用于Task10
   - Task10: 70epochs
    - 特征:相邻点RHS
    - 每个人样本的序列长度：100，模型保存名rnn6.pkl
    - 6层双向，隐藏层=400，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0
    - 准确率： Loss：(忘记录了...不过和之前单层差不多)
    - 测试: 5次测试均90%，每次都是8分成5.
    - 结论：双向LSTM和单向效果差不多。
    
3. 双向LSTM模型和1间隔RHS用于Task10
   - Task10: 70epochs
    - 特征:1间隔RHS
    - 每个人样本的序列长度：100，模型保存名rnn_2.pkl
    - 6层双向，隐藏层=400，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0
    - Train10 Loss: 0.018413, Train10 Acc: 0.994100, Eval Loss: 0.592419, Eval Acc: 0.883500
    - 测试: 5次测试，1次90%，4次100%。90%是8分成5，但是看上去比上面的相邻RHS好一点。
    - 结论: 1间隔RHS仍包含有区分Writers的重要信息，效果不比相邻RHS差。观察其训练过程在epochs为36时测试正确率已经达到90%，之后直到70epochs都没有什么提升，只是训练正确率慢慢达到99%以上，可能有过拟合现象。
   - Task10: 60epochs
    - 特征:1间隔RHS
    - 每个人样本的序列长度：100，模型保存名rnn_2.pkl
    - 6层双向，隐藏层=400，batch_size=1000，训练样本数1000，测试600，Adam学习率0.001，权重衰减0
    - epoch: 60, Train10 Loss: 0.145206, Train10 Acc: 0.953000, Eval Loss: 0.488143, Eval Acc: 0.857167
    - 测试: 5次测试，都是90%，问题都是是8分成5。
    - 结论: 加了decay的效果没有预想中的好，可能是epoch不太够？但是8分成5问题还是很明显，这次实验表明了1间隔RHS不能明显区分开5，8。但是突然好奇为什么5不会分为8，但是8会分成5？可能两个特征空间有交集，之后5的特征空间交集占比大，而8占比小？
   
    

 

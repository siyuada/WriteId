1.数据集/Data/
用于任务二：107分类的书写者鉴别任务。
/Data/Train/：共107个数据文件，每个文件以学号命名，包含该同学书写的300个汉字；
/Data/Validation/：共107个数据文件，每个文件以学号命名，包含该同学书写的100个汉字；
/Data/Validation_with_labels/含107个数据文件，与Validation内数据相同，但隐去学号；1个true_ids文件，包含按顺序排列的实际学号；
本文件夹与测试文件夹/Data/Test_with_labels/内的测试数据形式完全相同，但测试文件夹内学号不按顺序排列。要求同学利用本文件夹内数据调通测试接口test.py方可上交。

2.数据集/Data10/
用于任务一：10分类的书写者鉴别任务。
数据格式同1所述。

3.测试接口/TestCodes/test.py
（1）认真阅读代码内注释，按要求调通接口并提交；
（2）修改模型接口：‘somewhere’替换为保存模型的文件包，‘YourFunction’替换为你的模型名字；
（3）使用命令''python test.py --testfolder ../Validation_with_labels（测试目录） --num_class 10（或107，分类类别数）''运行test.py；
（4） 要求利用../Validation_with_labels/内数据调通测试接口test.py，除了上述要求修改的部分以外，其余代码请勿增删。

4.最终作业提交格式
-张三_李四_王五_书写者鉴别.zip
  -Codes/ #程序源代码
  -TestCodes/ #测试代码，含test.py
  -作业报告.pdf
每小组提交一份即可，并用独立段落说明成员贡献。
数据无需提交。
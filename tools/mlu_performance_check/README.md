# 性能测试环境检查与设置
	
## 文档修改记录

| 日期        | 版本 | 修改内容 |
| ----        | ---- | -------- |
| 2021.08.17  | v0.1 | 初版     |
|             |      |          |
|             |      |          |

## 1. CheckList
- 检查当前linux系统是否为Ubuntu或者Centos操作系统, 并输出对应系统版本号。
- 输出CPU型号，主频，核心数等相关信息。
- 检查CPU是否为performance模式，如果不是则报出WARNING。
- 检查CPU是否空闲，如果某个进程的CPU占用率超过5%，则会输出对应进程的PID和和COMMAND。
- 检查MLU是否空闲。
- 检查CPU的irqbalance是否开启，如果未开启则报出WARNING。
- 检查数据集是否在本地。
- 检查MLUTaskAccelerate，如果发现某张MLU的MLUTaskAccelerate未开启则报出WARNING(只针对MLU370及MLU290)。
- 检查MLULink，如果发现某张MLU卡有一个链路的status为Disable就会报出对应的卡号(只针对MLU370-X8及MLU290)。

## 2. SetList
- 检查CPU是否为performance模式，如果不是则将其设置为performance模式。
- 检查CPU的irqbalance是否开启，如果未开启则将其设置为running。
- 检查MLUTaskAccelerate，如果发现某张MLU的MLUTaskAccelerate未开启则开启(只针对MLU370及MLU290)。

## 3. 使用方法

- 检查当前环境是否符合性能测试的标准
> ./check_perf.sh path_to_datadir
- 将当前环境设置为符合性能测试的标准(需要root权限)
> sudo bash set_perf.sh

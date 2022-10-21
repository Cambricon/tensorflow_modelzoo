# Record time for benchmark

这里的代码用于记录tensorflow_modelzoo仓库中keras和estimator接口每一步的e2e运行时间。

## Installation
在当前目录下，运行：

```bash
pip install .
```

## Quick Start Guide
本工具提供了tensorflow2下，keras和estimator的每次迭代时间记录方式。使用前需要运行python脚本：
```python
from record-time import *
```
工具中提供了两个类与一个函数，`TimeHistoryRecord`，`TimeHook`和`write_json`，`TimeHistoryRecord` 记录keras接口的e2e时间，`TimeHook`记录estimator接口的e2e时间，write_json将记录的时间写入json。

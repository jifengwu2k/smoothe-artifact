import os.path
from typing import Any, Sequence
from pathlib import Path

import coverage
from trie import TrieNode, search_or_create

# 初始化 coverage 对象
cov = coverage.Coverage()

# 加载 .coverage 文件
cov.load()

# 获取覆盖数据
data = cov.get_data()

measured_files = data.measured_files()

root: TrieNode[str, Any] = TrieNode()

for filename in measured_files:
    parts: Sequence[str] = Path(filename).parts
    # os.path.join(*parts)
    search_or_create(root, parts)

# 遍历所有被测量的文件
for filename in data.measured_files():
    print(f"File: {filename}")

    # 获取该文件的覆盖行信息
    lines = data.lines(filename)
    if lines is not None:
        print(f"  Covered lines: {sorted(lines)}")

    print()

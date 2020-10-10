# MultiSMOTE
A multi-label approach for SMOTE algorithm

![APM](https://img.shields.io/apm/l/vim-mode)
[![Generic badge](https://img.shields.io/badge/python->=3-green.svg)](https://shields.io/)

---
Synthetic Minority Oversampling Technique which supports multi-label data.
The specific approach, resamples from the representative data that belongs only at the minority class.

Usage
---
```python
from multiSmote.multi_smote import MultiSmote as mlsmote

smote = mlsmote()
new_x, new_y = smote.multi_smote(X, y)
```
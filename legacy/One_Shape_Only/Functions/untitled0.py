# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:49:12 2026

@author: eleno
"""

import torch
nholes = 3
weights             = -torch.log(torch.rand(nholes).clamp_min(1e-12))
proportions         = weights/weights.sum()

weights             = 0.9 + 0.2 * torch.rand(3)
proportions         = weights/weights.sum()
print(proportions.sum())


print(weights)
print(proportions)
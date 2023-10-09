#!/usr/bin/env python
'''
test if normal perturbations change topk features
'''

import numpy as np
import torch
from openxai.explainers.perturbation_methods import NormalPerturbation
from openxai.experiment_utils import generate_mask

def test_mask():
    topk = 2
    x = torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    feature_metadata = ['c', 'c', 'c', 'c', 'c', 'c']

    mask = generate_mask(x, topk)
    assert mask.sum() == topk

    perturbation = NormalPerturbation('tabular', mean=0, std_dev=0.1, flip_percentage=0.3)
    x_perturbed = perturbation.get_perturbed_inputs(original_sample=x, feature_mask=mask, num_samples=1, 
        feature_metadata=feature_metadata)

    delta = x - x_perturbed
    assert torch.count_nonzero(delta).item() == topk
    return


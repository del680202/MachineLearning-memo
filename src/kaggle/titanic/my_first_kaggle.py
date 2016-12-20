#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import pandas as pd


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
rand_labels = (np.random.rand(len(test_data['PassengerId'])) > 0.5).astype(np.int32)

results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : rand_labels
})

results.to_csv(SCRIPT_PATH + "/submission1.csv", index=False)

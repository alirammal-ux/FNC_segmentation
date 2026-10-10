'''Applying a trained object classifier to a table of predicted objects. Objects touching the image border are never
judged (always kept): in the annotations they rarely have a counterpart, a pattern linked to the cut and not documented in
the annotation protocol, which the classifier should not learn (notebook 03, section 6).'''

import numpy as np

from evaluation.object_features import FEATURES


def keep_objects(model, table, threshold, features=FEATURES):
    '''True = the object is kept: P(true) ≥ threshold, or it touches the border.'''
    p_tp = model.predict_proba(table[features].to_numpy(np.float64))[:, 1]
    return (p_tp >= threshold) | table.border.to_numpy(bool)

# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
from typing import Optional
import optuna


def metric_dict_write(file, metric_dict, title=None, metric_sep_cl=False, trial: Optional[optuna.trial.Trial] = None):
    """Helper function to write ml efficiency metrics from a dict to an open file.

    Args:
        file: a file object return by `open`.
        metric_dict: a dict containing ml efficiency metrics, e.g. f1-score, accuracy.
        title: the title texts to be added at the first line before writing the metrics.
        metric_sep_cl: whether to change line after writing the metric name.
    """
    if title is not None:
        file.write(f'{title}\n\n')
    if trial is not None:
        file.write(f'params: {trial.params}\n')
    for key, value in metric_dict.items():
        if not metric_sep_cl:
            file.write(f'{key}: {value}\n')
        else:
            file.write(f'{key}:\n{value}\n\n')
    file.write(f"\n")


class Record_best_trials:
    def __init__(self, tune_path, multi_obj=True):
        self.tune_path = tune_path
        self.multi_obj = multi_obj

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        A function to be called after each trial finished to log the 
        best trials so far.
        """
        with open(os.path.join(self.tune_path, 'best_trials.txt'), 'w') as file:
            file.write(f"Best trials so far and tuning details:\n\n")
            if self.multi_obj:
                for best_trial in study.best_trials:
                    file.write(f"{best_trial} \n\n")
            else:
                file.write(f"{study.best_trial} \n\n")

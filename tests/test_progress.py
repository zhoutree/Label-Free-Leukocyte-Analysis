"""验证终端窄窗口只缩减进度条展示，不改变训练日志。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from leukocyte_analysis.training.progress import CompactRichProgressBar


class TestProgress(unittest.TestCase):
    """只检查显示层的缩写，不更改记录到 TensorBoard 的指标。"""

    def test_rich_progress_adapts_metrics_to_terminal_width(self) -> None:
        """窄窗口逐级隐藏指标，同时保留进度且不触及日志键名。"""
        reported = {"step_weighted_loss": 0.4, "val_loss": 0.2,
                    "val_macro_f1": 0.8, "v_num": 1}
        with patch("leukocyte_analysis.training.progress.RichProgressBar.get_metrics",
                   return_value=reported):
            progress = CompactRichProgressBar(refresh_rate=5)
            progress._console = SimpleNamespace(width=160)
            self.assertEqual(progress.get_metrics(None, None),
                             {"loss": 0.4, "val_loss": 0.2, "val_f1": 0.8})
            progress._console.width = 125
            self.assertEqual(progress.get_metrics(None, None), {"loss": 0.4, "val_f1": 0.8})
            progress._console.width = 95
            self.assertEqual(progress.get_metrics(None, None), {"loss": 0.4})
            progress._console.width = 60
            self.assertEqual(progress.get_metrics(None, None), {})
            self.assertEqual(str(progress.configure_columns(None)[-1].render(None)), "")

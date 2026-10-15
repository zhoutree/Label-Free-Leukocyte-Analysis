"""让 Lightning 的 Rich 进度条按终端宽度收起次要信息。"""

import sys

from lightning.pytorch.callbacks import RichProgressBar
from rich import get_console
from rich.progress import BarColumn, MofNCompleteColumn, ProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Column
from rich.text import Text


def prepare_console_output() -> None:
    """GBK 等终端编码不能表示 Rich 符号时，以替代字符显示而不中断训练。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


class _ResponsiveTimeColumn(ProgressColumn):
    """窗口变窄时隐藏预计剩余时间；每次重绘都读取实时宽度。"""

    def __init__(self, progress_bar: "CompactRichProgressBar") -> None:
        """引用外层进度条，以便重绘时读取当前终端宽度。"""
        super().__init__()
        self.progress_bar = progress_bar
        self.time_column = TimeRemainingColumn()

    def render(self, task) -> Text:
        """窄于 80 列时隐藏预计剩余时间。"""
        if self.progress_bar.terminal_width < 80:
            return Text("")
        return self.time_column(task)


class CompactRichProgressBar(RichProgressBar):
    """复用 Rich 排版，宽窗口显示指标，窄窗口优先保留批次进度。"""

    @property
    def terminal_width(self) -> int:
        """返回 Rich 控制台的实时列数。"""
        return self._console.width if self._console is not None else get_console().width

    def configure_columns(self, trainer) -> list:
        """让横向进度条占用其余空间，而不固定宽度。"""
        # 横条占用剩余空间；其它列按当前窗口宽度决定是否显示。
        return [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, table_column=Column(ratio=1)),
            MofNCompleteColumn(),
            _ResponsiveTimeColumn(self),
        ]

    def get_metrics(self, trainer, pl_module) -> dict:
        """按窗口宽度逐级缩减终端指标，不更改日志指标名。"""
        metrics = super().get_metrics(trainer, pl_module)
        width = self.terminal_width
        if width < 80:
            return {}
        if width < 115:
            labels = (("step_weighted_loss", "loss"),)
        elif width < 145:
            labels = (("step_weighted_loss", "loss"), ("val_macro_f1", "val_f1"))
        else:
            labels = (
                ("step_weighted_loss", "loss"),
                ("val_loss", "val_loss"),
                ("val_macro_f1", "val_f1"),
            )
        return {short: metrics[name] for name, short in labels if name in metrics}

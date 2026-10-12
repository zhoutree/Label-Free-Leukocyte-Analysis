"""Lightning 回调：逐轮保存验证错分，并统计同一图块重复出错的次数。"""

import csv
from collections import Counter
from pathlib import Path

import lightning as L

from ..evaluation.error_records import (
    ERROR_FIELDS,
    FREQUENCY_FIELDS,
    misclassified_rows,
    write_csv,
)


class ValidationErrorLogger(L.Callback):
    """仅记录验证集；不参与损失、模型选择或反向传播。"""

    def __init__(self, dataset, output_dir: Path) -> None:
        super().__init__()
        if not dataset.include_index:
            raise ValueError("验证集必须返回 packed_index")
        self.dataset = dataset
        self.output_dir = output_dir
        self._current_errors: list[dict] = []
        self._history: dict[int, dict] = {}
        self._completed_epochs = 0
        # 续训时从已有逐轮 CSV 恢复历史；旧检查点无需包含此回调状态。
        for path in sorted(output_dir.glob("epoch_*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self._completed_epochs += 1
            for row in rows:
                index = int(row["packed_index"])
                row["packed_index"] = index
                entry = self._history.setdefault(index, {
                    "source": row, "epochs": [], "predictions": Counter(),
                })
                entry["epochs"].append(int(row["epoch"]))
                entry["predictions"][row["predicted_label"]] += 1

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._current_errors = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if trainer.sanity_checking or outputs is None:
            return
        fields = ("packed_index", "true_index", "predicted_index", "confidence")
        if len({len(outputs[name]) for name in fields}) != 1:
            raise ValueError("验证批次的错分索引与预测数量不一致")
        self._current_errors.extend(dict(zip(fields, values)) for values in zip(
            *(outputs[name] for name in fields)
        ))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch  # Lightning 使用从 0 开始的轮号。
        rows = misclassified_rows(self.dataset, self._current_errors, "val", epoch)
        path = self.output_dir / f"epoch_{epoch:03d}.csv"
        if path.exists():
            raise FileExistsError(f"不覆盖已有验证错分记录：{path}")
        write_csv(path, rows, ERROR_FIELDS)
        self._completed_epochs += 1

        for row in rows:
            index = row["packed_index"]
            entry = self._history.setdefault(index, {
                "source": row, "epochs": [], "predictions": Counter(),
            })
            entry["epochs"].append(epoch)
            entry["predictions"][row["predicted_label"]] += 1

        # 每轮更新一次；提前停止或 Ctrl+C 后仍能查看已完成轮次。
        summary = []
        for index, entry in self._history.items():
            source = entry["source"]
            summary.append({
                **{field: source[field] for field in (
                    "packed_index", "file", "donor", "row", "col",
                    "source_label", "true_label",
                )},
                "wrong_epochs": len(entry["epochs"]),
                "evaluated_epochs": self._completed_epochs,
                "wrong_rate": len(entry["epochs"]) / self._completed_epochs,
                "epochs": ";".join(map(str, entry["epochs"])),
                "predicted_labels": ";".join(
                    f"{label}:{count}" for label, count in
                    sorted(entry["predictions"].items())
                ),
            })
        summary.sort(key=lambda row: (-row["wrong_epochs"], row["packed_index"]))
        write_csv(self.output_dir / "frequency.csv", summary, FREQUENCY_FIELDS)

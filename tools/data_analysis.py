"""CSV/XLSX 数据读取与基础统计分析。"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# 受限账户可能无法写入用户主目录，把 Matplotlib 缓存放到系统临时目录。
_matplotlib_cache = Path(tempfile.gettempdir()) / "tiyan-zhinao-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import pandas as pd
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


class DataAnalysisError(ValueError):
    """数据文件无法读取或无法分析。"""


@dataclass(frozen=True)
class DataAnalysisResult:
    """一次数据分析的摘要结果。"""

    dataframe: pd.DataFrame
    rows: int
    columns: int
    duplicate_rows: int
    numeric_summary: pd.DataFrame
    missing_summary: pd.DataFrame


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """根据扩展名读取 CSV 或 XLSX 数据。"""

    if not file_bytes:
        raise DataAnalysisError("上传文件为空，请重新选择文件。")

    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            dataframe = _read_csv_with_fallback(file_bytes)
        elif suffix in {".xlsx", ".xlsm"}:
            dataframe = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        else:
            raise DataAnalysisError("仅支持 .csv、.xlsx 或 .xlsm 文件。")
    except DataAnalysisError:
        raise
    except ImportError as exc:
        raise DataAnalysisError(
            "读取 Excel 需要 openpyxl，请先安装 requirements.txt 中的依赖。"
        ) from exc
    except Exception as exc:
        raise DataAnalysisError(f"文件读取失败，请检查文件格式和内容。详细信息：{exc}") from exc

    if dataframe.empty:
        raise DataAnalysisError("文件读取成功，但其中没有数据行。")

    if len(dataframe.columns) == 0:
        raise DataAnalysisError("文件中没有可识别的数据列。")

    return dataframe


def analyze_dataframe(dataframe: pd.DataFrame) -> DataAnalysisResult:
    """生成行数、缺失值和数值列描述统计。"""

    if dataframe.empty:
        raise DataAnalysisError("数据为空，无法进行统计分析。")

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        numeric_data = dataframe[numeric_columns]
        numeric_summary = pd.DataFrame(
            {
                "字段": numeric_columns,
                "非空数量": [int(numeric_data[column].count()) for column in numeric_columns],
                "均值": [numeric_data[column].mean() for column in numeric_columns],
                "标准差": [
                    numeric_data[column].std(ddof=1) for column in numeric_columns
                ],
                "最小值": [numeric_data[column].min() for column in numeric_columns],
                "最大值": [numeric_data[column].max() for column in numeric_columns],
            }
        )
    else:
        numeric_summary = pd.DataFrame(
            columns=["字段", "非空数量", "均值", "标准差", "最小值", "最大值"]
        )

    missing_counts = dataframe.isna().sum()
    missing_rows = [
        {
            "字段": str(column),
            "缺失数量": int(count),
            "缺失比例": f"{count / len(dataframe):.1%}",
        }
        for column, count in missing_counts.items()
        if int(count) > 0
    ]
    missing_summary = pd.DataFrame(
        missing_rows, columns=["字段", "缺失数量", "缺失比例"]
    )

    return DataAnalysisResult(
        dataframe=dataframe,
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        duplicate_rows=int(dataframe.duplicated().sum()),
        numeric_summary=numeric_summary,
        missing_summary=missing_summary,
    )


def create_numeric_histogram(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 20,
) -> Figure:
    """为指定数值列创建直方图。"""

    if column not in dataframe.columns:
        raise DataAnalysisError(f"数据中不存在字段“{column}”。")
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise DataAnalysisError(f"字段“{column}”不是数值列，无法绘制直方图。")

    values = dataframe[column].dropna()
    if values.empty:
        raise DataAnalysisError(f"字段“{column}”没有可用于绘图的数据。")
    if bins <= 0:
        raise DataAnalysisError("直方图分组数量必须是正整数。")

    figure = Figure(figsize=(8.0, 4.2), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.hist(values, bins=bins, color="#0F766E", edgecolor="white", alpha=0.9)
    axis.set_title(f"{column} 分布", fontsize=13, pad=12)
    axis.set_xlabel(column)
    axis.set_ylabel("频数")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", alpha=0.2, linestyle="--")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    """依次尝试常见中文编码读取 CSV，避免 GBK 文件直接报错。"""

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: 编码不匹配")
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    detail = "；".join(errors[-2:])
    raise DataAnalysisError(f"CSV 读取失败，请检查编码和分隔符。详细信息：{detail}")

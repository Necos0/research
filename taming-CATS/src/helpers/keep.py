"""数値保持（<KEEP> 制御トークン）用の共通ヘルパ。

本研究の提案手法では、原文中の「消失させてはならない数値」を抽出し、
制御トークン <KEEP=value> として入力に付与する。数値の抽出・整形・保持率の
計算は、学習/推論データへのメトリック付与（add_keep_metric.py）と、評価時の
Metrics.compute_metrics() の双方で **同一のロジック** を使う必要があるため、
ここに一元化する。
"""

import re

# 整数・小数・桁区切り(1,234)・パーセント等の数値本体を抽出する。
#   例: "0.2 %", "1,234", "1995", "0.3-0.7" → ["0.2", "1,234", "1995", "0.3", "0.7"]
_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

# 数値を持たない事例の番兵（<KEEP=none> として学習させ、制約なしを学ばせる）
NO_NUMBERS = "none"


def extract_numbers(text):
    """text から数値トークンを出現順・重複排除で抽出して返す（list[str]）。"""
    if not isinstance(text, str):
        return []
    seen = []
    for match in _NUM_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def format_keep(numbers):
    """抽出した数値リストを <KEEP=...> トークンに埋める文字列へ整形する。"""
    return ", ".join(numbers) if numbers else NO_NUMBERS


def keep_value_from_source(source_text):
    """原文から KEEP トークン値（保持すべき数値の整形済み文字列）を作る。

    推論・評価用: 「原文の全数値を保持せよ」という指示のタグ値。
    """
    return format_keep(extract_numbers(source_text))


def keep_value_from_pair(source_text, target_text):
    """学習用の KEEP トークン値: 原文の数値のうち正解文が実際に保持しているもの。

    タグと教師信号を必ず整合させるため（FKGL タグが「正解文の実際の FKGL 値」を
    使うのと同じ原則）、正解が保持していない数値はタグに入れない。
    1つも保持されていなければ NO_NUMBERS（保持制約なしの教師として使う）。
    """
    target_numbers = set(extract_numbers(target_text))
    retained = [n for n in extract_numbers(source_text) if n in target_numbers]
    return format_keep(retained)


def keep_retention(source_text, text):
    """原文の数値のうち text にも現れる割合（数値保持率）を返す。

    - 原文に数値が無い場合は 1.0（保持すべき数値が無い＝満点）。
    - 評価時に reference/prediction の保持率を比べる（MAE 等）ための機械的指標。
    """
    source_numbers = extract_numbers(source_text)
    if not source_numbers:
        return 1.0
    text_numbers = set(extract_numbers(text))
    kept = sum(1 for number in source_numbers if number in text_numbers)
    return kept / len(source_numbers)

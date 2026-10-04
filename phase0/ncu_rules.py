"""阶段 0 · 把 ncu 的官方规则输出整理成报告。

为什么需要它：`ncu --set full` 输出的 INF / OPT / WRN 是 NVIDIA 内置规则引擎的结论，
**带量化收益**（Est. Speedup），应该作为每次优化的起点。
但它夹在几十个 section 里，肉眼翻很累；本脚本把它抽出来、按收益排序、写成 Markdown。

它在工作流里的位置（notes/09 §0.1）：**第 1 层「官方 rules 找症状」的入口**。

运行::

    # 只看规则清单（打印到屏幕）
    python phase0/ncu_rules.py phase0/cuda/sgemm.exe -k sgemm_reg2 --skip 3

    # 同时写成报告
    python phase0/ncu_rules.py phase0/cuda/sgemm.exe -k sgemm_reg2 --skip 3 \
        --out phase0/reports/sgemm_reg2_rules.md

    # 采全部 kernel（不写 -k），只采 1 次
    python phase0/ncu_rules.py phase0/cuda/vecadd.exe --count 1
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Windows 上 ncu 是 .bat；subprocess 直接给 "ncu" 会找不到，必须先解析成完整路径
NCU_FALLBACK = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.1\ncu.bat"

SECTION_RE = re.compile(r"^\s+Section:\s*(.+?)\s*$")
RULE_RE = re.compile(r"^(\s+)(INF|OPT|WRN)\s+(.*)$")
SPEEDUP_RE = re.compile(r"Est\.\s*(?:Local\s+)?Speedup:\s*([\d.]+)\s*%")


def ncu_path() -> str:
    found = shutil.which("ncu")
    if found:
        return found
    if pathlib.Path(NCU_FALLBACK).exists():
        return NCU_FALLBACK
    raise FileNotFoundError("找不到 ncu（既不在 PATH，也不在默认安装路径）")


def run_ncu(exe: str, kernel: str | None, skip: int, count: int) -> tuple[str, list[str]]:
    cmd = [ncu_path(), "--set", "full"]
    if kernel:
        cmd += ["-k", f"regex:{kernel}"]
    cmd += ["--launch-skip", str(skip), "--launch-count", str(count), exe]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)
    return (proc.stdout or "") + (proc.stderr or ""), cmd


def parse_rules(text: str) -> list[dict]:
    """从 ncu 输出里抽出所有规则，并关联它所在的 section。"""
    lines = text.splitlines()
    rules: list[dict] = []
    section = "?"
    i = 0
    while i < len(lines):
        m = SECTION_RE.match(lines[i])
        if m:
            section = m.group(1)
            i += 1
            continue

        m = RULE_RE.match(lines[i])
        if not m:
            i += 1
            continue

        indent = len(m.group(1))
        severity = m.group(2)
        body = [m.group(3).strip()]

        # 续行：缩进更深，且不是新的规则/section
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if SECTION_RE.match(nxt) or RULE_RE.match(nxt):
                break
            cur_indent = len(nxt) - len(nxt.lstrip())
            if nxt.strip() and cur_indent >= indent + 4:
                body.append(nxt.strip())
                j += 1
                continue
            if not nxt.strip():
                # 空行：若后面紧跟的仍是更深的缩进，就当作段落继续
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if (k < len(lines) and not SECTION_RE.match(lines[k])
                        and not RULE_RE.match(lines[k])
                        and (len(lines[k]) - len(lines[k].lstrip())) >= indent + 4):
                    body.append("")
                    j += 1
                    continue
            break

        joined = " ".join(x for x in body if x).strip()
        sp = SPEEDUP_RE.search(joined)
        # 文本开头的 "Est. Speedup: xx%" 已经有独立列了，去掉避免重复
        lead = SPEEDUP_RE.match(joined)
        text_clean = joined[lead.end():].strip() if lead else joined
        rules.append({
            "section": section,
            "severity": severity,
            "speedup": float(sp.group(1)) if sp else None,
            "text": text_clean,
        })
        i = j
    return rules


def to_markdown(rules: list[dict], exe: str, kernel: str | None, skip: int,
                count: int, cmd: list[str]) -> str:
    today = datetime.date.today().isoformat()
    ranked = sorted([r for r in rules if r["speedup"] is not None],
                    key=lambda r: -r["speedup"])
    notes = [r for r in rules if r["speedup"] is None]

    out: list[str] = []
    out.append(f"# ncu 官方规则报告 · {kernel or '(全部 kernel)'}")
    out.append("")
    out.append(f"- 采集时间：{today}")
    out.append(f"- 被测程序：`{exe}`")
    out.append(f"- 复现命令：`{' '.join(cmd)}`")
    out.append(f"- 规则总数：{len(rules)}（其中 {len(ranked)} 条带收益估计）")
    out.append("")
    out.append("> 这是 NVIDIA 规则引擎的判断，属于**症状**层面。"
               "因果链解释与手段选择见 `notes/09` 的杠杆表。")
    out.append("")

    out.append("## 按预计收益排序")
    out.append("")
    if ranked:
        out.append("| # | 类型 | 预计收益 | 官方结论 | 出处 |")
        out.append("|---|---|---|---|---|")
        for i, r in enumerate(ranked, 1):
            text = r["text"].replace("|", "/")
            out.append(f"| {i} | {r['severity']} | **{r['speedup']:.2f} %** | {text} | {r['section']} |")
    else:
        out.append("（没有带收益估计的规则）")
    out.append("")

    if notes:
        out.append("## 无收益估计的其它规则")
        out.append("")
        for r in notes:
            out.append(f"- **{r['severity']}** [{r['section']}] {r['text']}")
        out.append("")

    out.append("## 分 section 明细")
    out.append("")
    by_section: dict[str, list[dict]] = {}
    for r in rules:
        by_section.setdefault(r["section"], []).append(r)
    for sec, items in by_section.items():
        out.append(f"### {sec}")
        out.append("")
        for r in items:
            tag = r["severity"]
            if r["speedup"] is not None:
                tag += f"（Est. Speedup {r['speedup']:.2f} %）"
            out.append(f"- **{tag}** {r['text']}")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取 ncu 官方规则并整理成报告")
    ap.add_argument("exe", help="被测可执行文件（相对仓库根目录）")
    ap.add_argument("-k", "--kernel", default=None, help="kernel 名（正则）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 次匹配的 launch")
    ap.add_argument("--count", type=int, default=1, help="采集几个 launch")
    ap.add_argument("--out", default=None, help="写成 Markdown 的路径")
    args = ap.parse_args()

    text, cmd = run_ncu(args.exe, args.kernel, args.skip, args.count)
    rules = parse_rules(text)
    if not rules:
        print("没解析到任何规则。原始输出尾部：", file=sys.stderr)
        print("\n".join(text.splitlines()[-20:]), file=sys.stderr)
        sys.exit(1)

    ranked = sorted([r for r in rules if r["speedup"] is not None],
                    key=lambda r: -r["speedup"])
    print(f"共 {len(rules)} 条规则，{len(ranked)} 条带收益估计\n")
    print(f"  {'类型':<5} {'收益':>8}  官方结论")
    for r in ranked:
        text_line = r["text"][:96]
        print(f"  {r['severity']:<5} {r['speedup']:>7.2f}%  {text_line}")
    rest = [r for r in rules if r["speedup"] is None]
    if rest:
        print(f"\n  另有 {len(rest)} 条无收益估计的提示（INF/WRN）")

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            to_markdown(rules, args.exe, args.kernel, args.skip, args.count, cmd),
            encoding="utf-8")
        print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()

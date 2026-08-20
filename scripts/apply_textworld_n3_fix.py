#!/usr/bin/env python
"""N3 一行真修:PddlEnv.close() 释放 fast_downward 的 dlopen 句柄。

根因(docs/infra/LANDMINES.md L2/N3):每个 PddlEnv 构造时
fast_downward.load_lib() 复制 libdownward.so 到临时目录并 dlopen 私有副本,
但 close_lib 全仓零调用 —— 长寿命 AlfWorld/AgentGym 进程的 mmap 区只增不减,
最终 "failed to map segment"。textworld.core.Environment.__del__ 会调
close(),故在 PddlEnv 覆写 close() 后,显式关闭与 GC 两条路都能回收。

幂等:重复运行不重复打;环境重建(pip 重装 textworld)后重跑本脚本即可。

用法:
  python scripts/apply_textworld_n3_fix.py \
      /projects_vol/.../envs/duet/lib/python3.11/site-packages [more...]
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "DUET N3 fix"

PATCH = '''
    def close(self) -> None:
        # DUET N3 fix (2026-08-13): each PddlEnv dlopens a private copy of
        # libdownward.so (fast_downward.load_lib) that is never released, so
        # long-lived AlfWorld servers exhaust the process mmap budget
        # ("libdownward.so: failed to map segment"). Release the handle
        # exactly once; base Environment.__del__ routes GC through close().
        # Applied by scripts/apply_textworld_n3_fix.py (idempotent).
        lib = getattr(self, "downward_lib", None)
        if lib is not None:
            try:
                fast_downward.close_lib(lib)
            except Exception:
                pass
            finally:
                self.downward_lib = None
        super().close()
'''

ANCHOR = "        super().__init__(infos)\n        self.downward_lib = fast_downward.load_lib()\n"


def apply(site_packages: Path) -> str:
    target = site_packages / "textworld" / "envs" / "pddl" / "pddl.py"
    if not target.is_file():
        return f"SKIP {target} (missing)"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        return f"OK   {target} (already patched)"
    if ANCHOR not in text:
        return f"FAIL {target} (anchor not found — upstream changed, patch manually)"
    text = text.replace(ANCHOR, ANCHOR + PATCH, 1)
    backup = target.with_suffix(".py.n3fix.bak")
    if not backup.exists():
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.write_text(text, encoding="utf-8")
    # 编译自检:打完必须仍可 import
    import py_compile

    py_compile.compile(str(target), doraise=True)
    return f"DONE {target}"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    failed = False
    for arg in sys.argv[1:]:
        result = apply(Path(arg))
        print(result)
        failed |= result.startswith("FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

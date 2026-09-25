"""前端端到端渲染验证（无头 Edge + Chrome DevTools Protocol）。

为什么需要它
------------
命令行 `--headless --screenshot` 只能渲染**未登录**页面，无法验证核心的识别页、历史页、
个人中心 —— 这些页面需要 localStorage 里有 JWT。本脚本通过 CDP 注入 token，
再逐页截图，从而覆盖**登录后**的真实渲染。

同时也用于 M5-3 端到端回归与答辩前自检。

用法
----
    # 1) 先启动后端（单端口托管前端构建产物）
    python -m uvicorn --app-dir backend app.main:app --port 8000
    # 2) 再跑本脚本
    python scripts/verify_frontend.py

产出：`.cache/shots/e2e-*.png`，退出码 0 表示全部页面截图成功。

依赖：`websockets`（随 uvicorn[standard] 一并安装）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import shutil
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def find_browser() -> Path:
    for c in EDGE_CANDIDATES:
        if c.is_file():
            return c
    raise SystemExit("未找到 Edge / Chrome，无法进行渲染验证")


# ---------------------------------------------------------------------------
# 后端交互
# ---------------------------------------------------------------------------
def api_post(base: str, path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def ensure_user_and_token(base: str, username: str, password: str) -> tuple[str, int]:
    """注册（若已存在则直接登录）并返回 (token, record_count)。"""
    try:
        api_post(base, "/api/v1/auth/register",
                 {"username": username, "email": f"{username}@example.com", "password": password})
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
    token = api_post(base, "/api/v1/auth/login",
                     {"username": username, "password": password})["data"]["access_token"]
    return token


def seed_one_record(base: str, token: str, image: Path) -> int:
    """上传一张图，保证历史页/个人中心有数据可展示。"""
    boundary = "----verify" + "".join(random.choices(string.ascii_letters + string.digits, k=12))
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{image.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        base + "/api/v1/predict", data=body, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    return int(payload["data"]["record_id"])


# ---------------------------------------------------------------------------
# CDP
# ---------------------------------------------------------------------------
def wait_devtools(port: int, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=3) as r:
                targets = json.loads(r.read())
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise SystemExit(f"无法连接 DevTools 调试端口 {port}（{last}）")


class CdpSession:
    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def send(self, method: str, **params) -> dict:
        self._id += 1
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=60))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败：{msg['error']}")
                return msg.get("result", {})

    async def evaluate(self, expression: str) -> object:
        res = await self.send("Runtime.evaluate", expression=expression, returnByValue=True)
        return res.get("result", {}).get("value")


async def render_pages(
    ws_url: str, base: str, token: str, user: dict, out_dir: Path, width: int, height: int
) -> list[tuple[str, str, int]]:
    """注入 token 后逐页截图，返回 [(路由, 文件, 字节数)]。"""
    import websockets

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, int]] = []
    snapshot = {
        "flowers_token": token,
        "flowers_user": json.dumps(user, ensure_ascii=False),
    }

    pages = [
        ("/recognize", "e2e-1-recognize"),
        ("/history", "e2e-2-history"),
        ("/profile", "e2e-3-profile"),
        ("/flowers/73", "e2e-4-flower-detail"),
        ("/models", "e2e-5-models"),
    ]

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        cdp = CdpSession(ws)
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        await cdp.send(
            "Emulation.setDeviceMetricsOverride",
            width=width, height=height, deviceScaleFactor=1, mobile=False,
        )

        # 先到同源页面写入 localStorage，之后 SPA 才能读到 token
        await cdp.send("Page.navigate", url=f"{base}/login")
        await asyncio.sleep(3)
        for key, value in snapshot.items():
            await cdp.evaluate(f"localStorage.setItem({json.dumps(key)}, {json.dumps(value)})")
        stored = await cdp.evaluate("localStorage.getItem('flowers_token') ? 'ok' : 'missing'")
        print(f"  localStorage 注入：{stored}")

        for route, name in pages:
            await cdp.send("Page.navigate", url=f"{base}{route}")
            # 等 SPA 路由 + 接口请求完成
            await asyncio.sleep(6)
            shot = await cdp.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            data = base64.b64decode(shot["data"])
            path = out_dir / f"{name}.png"
            path.write_bytes(data)
            url_now = await cdp.evaluate("location.pathname")
            results.append((route, f"{name}.png", len(data)))
            print(f"  {route:18s} → 渲染于 {url_now:18s} {len(data) / 1024:7.1f} KB")

    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="前端端到端渲染验证（无头 Edge + CDP）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", dest="base")
    ap.add_argument("--debug-port", type=int, default=9222, dest="port")
    ap.add_argument("--out-dir", default=str(ROOT / ".cache" / "shots"), dest="out_dir")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--keep-browser", action="store_true", dest="keep")
    args = ap.parse_args(argv)

    base = args.base.rstrip("/")
    out_dir = Path(args.out_dir)
    profile = ROOT / ".cache" / "edge-profile-e2e"
    shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True, exist_ok=True)

    print("=== 1. 准备账号与数据 ===")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    username = f"e2e_{suffix}"
    password = "E2eVerify123"
    token = ensure_user_and_token(base, username, password)
    print(f"  账号 {username} 已就绪，token 长度 {len(token)}")

    image = ROOT / "data" / "flowers-102" / "jpg" / "image_00001.jpg"
    if image.is_file():
        record_id = seed_one_record(base, token, image)
        print(f"  已生成 1 条识别记录（record_id={record_id}）")
    else:
        print(f"  ⚠ 未找到 {image}，历史与统计页可能为空")

    user = {"id": 0, "username": username, "email": f"{username}@example.com",
            "created_at": ""}
    # 取真实用户信息（个人中心要显示）
    req = urllib.request.Request(
        base + "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        user = json.loads(r.read())["data"]

    print("\n=== 2. 启动无头浏览器 ===")
    browser = find_browser()
    proc = subprocess.Popen(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            "--disable-crash-reporter",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={args.port}",
            f"--window-size={args.width},{args.height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = wait_devtools(args.port)
        print(f"  DevTools 就绪：{ws_url[:60]}…")
        print("\n=== 3. 注入登录态并逐页截图 ===")
        results = asyncio.run(
            render_pages(ws_url, base, token, user, out_dir, args.width, args.height)
        )
    finally:
        if not args.keep:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("\n=== 4. 汇总 ===")
    ok = 0
    for route, name, size in results:
        good = size > 5000
        ok += 1 if good else 0
        print(f"  {'✓' if good else '✗'} {route:18s} {name:24s} {size / 1024:7.1f} KB")
    print(f"\n页面截图：{ok}/{len(results)} 成功，输出目录 {out_dir}")
    return 0 if ok == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())

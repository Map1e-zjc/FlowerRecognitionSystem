"""后端接口冒烟测试（对**运行中的**服务发起真实 HTTP 请求）。

用途
----
* M3-3/M3-4 开发期快速回归；
* M5-3 端到端回归与答辩前自检。

用法
----
    # 先启动后端（项目根目录）
    python -m uvicorn --app-dir backend app.main:app --port 8000
    # 再跑冒烟
    python scripts/smoke_backend.py
    python scripts/smoke_backend.py --base-url http://127.0.0.1:8000 --image data/flowers-102/jpg/image_00001.jpg

退出码：全部通过为 0，有失败为 1（可直接用于 CI / 验收脚本）。
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL = "[PASS]", "[FAIL]"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  → {detail}" if detail else ""))
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="后端接口冒烟测试")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", dest="base_url")
    ap.add_argument("--image", default=None, help="用于识别测试的图片（默认取数据集第一张）")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    base = args.base_url.rstrip("/")
    api = f"{base}/api/v1"

    image_path = Path(args.image) if args.image else ROOT / "data" / "flowers-102" / "jpg" / "image_00001.jpg"
    if not image_path.is_absolute():
        image_path = ROOT / image_path

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    username = f"smoke_{suffix}"
    email = f"{username}@example.com"
    password = "Smoke12345"

    with httpx.Client(timeout=args.timeout) as c:
        # ---------------- 1) 健康检查 ----------------
        print("\n=== 1. 健康检查 /healthz ===")
        try:
            r = c.get(f"{base}/healthz")
            data = r.json().get("data", {})
            check("GET /healthz 返回 200", r.status_code == 200, f"HTTP {r.status_code}")
            check("数据库可用", data.get("database") == "ok", str(data.get("database")))
            check("百科数据已导入（102）", data.get("flowers") == 102, f"flowers={data.get('flowers')}")
            check("模型已加载", bool(data.get("model_loaded")),
                  f"model={data.get('model_name')} err={data.get('model_error')}")
            check("GPU 可用", bool(data.get("gpu_available")), str(data.get("gpu_name")))
        except Exception as exc:  # noqa: BLE001
            check("GET /healthz", False, f"{type(exc).__name__}: {exc}")
            print("\n服务未启动？请先运行：python -m uvicorn --app-dir backend app.main:app --port 8000")
            return 1

        # ---------------- 2) 未登录保护 ----------------
        print("\n=== 2. 未登录访问受保护接口 ===")
        r = c.post(f"{api}/predict")
        check("POST /predict 未登录返回 401", r.status_code == 401, f"HTTP {r.status_code}")
        check("未登录提示为中文", "登录" in r.json().get("message", ""), r.json().get("message", ""))
        r = c.get(f"{api}/auth/me")
        check("GET /auth/me 未登录返回 401", r.status_code == 401, f"HTTP {r.status_code}")

        # ---------------- 3) 注册 ----------------
        print("\n=== 3. 注册 ===")
        r = c.post(f"{api}/auth/register",
                   json={"username": username, "email": email, "password": password})
        check("注册成功", r.status_code == 200, f"HTTP {r.status_code} {r.text[:120]}")
        user_id = (r.json().get("data") or {}).get("id")

        r = c.post(f"{api}/auth/register",
                   json={"username": username, "email": email, "password": password})
        check("重复用户名返回 409", r.status_code == 409, f"HTTP {r.status_code}")

        r = c.post(f"{api}/auth/register",
                   json={"username": f"{username}x", "email": "not-an-email", "password": password})
        check("非法邮箱返回 422", r.status_code == 422, f"HTTP {r.status_code}")

        r = c.post(f"{api}/auth/register",
                   json={"username": f"{username}y", "email": f"{username}y@example.com",
                         "password": "short"})
        check("弱密码返回 422", r.status_code == 422, f"HTTP {r.status_code}")

        # ---------------- 4) 登录 ----------------
        print("\n=== 4. 登录 ===")
        r = c.post(f"{api}/auth/login", json={"username": username, "password": password})
        check("用户名登录成功", r.status_code == 200, f"HTTP {r.status_code}")
        token = (r.json().get("data") or {}).get("access_token", "")
        check("拿到 access_token", bool(token), f"len={len(token)}")

        r = c.post(f"{api}/auth/login", json={"username": email, "password": password})
        check("邮箱也可登录", r.status_code == 200, f"HTTP {r.status_code}")

        r = c.post(f"{api}/auth/login", json={"username": username, "password": "WrongPass123"})
        check("密码错误返回 401", r.status_code == 401, f"HTTP {r.status_code}")
        check("错误提示为中文且不泄露用户是否存在",
              r.json().get("message") == "用户名或密码错误", r.json().get("message", ""))

        auth = {"Authorization": f"Bearer {token}"}

        r = c.get(f"{api}/auth/me", headers=auth)
        check("GET /auth/me 返回当前用户", r.status_code == 200 and
              (r.json().get("data") or {}).get("id") == user_id, f"HTTP {r.status_code}")

        r = c.get(f"{api}/auth/me", headers={"Authorization": "Bearer bad.token.here"})
        check("伪造令牌返回 401", r.status_code == 401, f"HTTP {r.status_code}")

        # ---------------- 5) 识别 ----------------
        print("\n=== 5. 识别接口 ===")
        if not image_path.exists():
            check("测试图片存在", False, str(image_path))
        else:
            with open(image_path, "rb") as fh:
                r = c.post(f"{api}/predict", headers=auth,
                           files={"file": (image_path.name, fh, "image/jpeg")})
            ok = r.status_code == 200
            check("识别成功", ok, f"HTTP {r.status_code} {r.text[:160]}")
            if ok:
                d = r.json()["data"]
                preds = d["predictions"]
                check("返回 Top-5", len(preds) == 5, f"len={len(preds)}")
                check("置信度降序", all(preds[i]["confidence"] >= preds[i + 1]["confidence"]
                                      for i in range(len(preds) - 1)),
                      str([p["confidence"] for p in preds]))
                # 注意：softmax 是在**全部 102 类**上归一化的，Top-5 只覆盖其中 5 类，
                # 因此 Top-5 置信度之和必然 ≤ 1（模型越自信就越接近 Top-1 的值），
                # 不能断言"之和 ≈ 1"。
                conf_sum = sum(p["confidence"] for p in preds)
                check("置信度均在 (0,1] 且 Top-5 之和 ≤ 1",
                      0 < conf_sum <= 1.0001 and all(0 < p["confidence"] <= 1 for p in preds),
                      f"sum(top5)={conf_sum:.4f}，其余 {1 - conf_sum:.4f} 分散在另外 97 个类")
                check("Top-1 在 Top-5 中占绝对多数",
                      preds[0]["confidence"] / conf_sum > 0.9,
                      f"占比={preds[0]['confidence'] / conf_sum:.1%}")
                check("含中文名", all(p["name_cn"] for p in preds),
                      " / ".join(p["name_cn"] for p in preds[:3]))
                check("推理耗时可接受（<300ms，服务启动时已预热）",
                      d["latency_ms"] < 300, f"{d['latency_ms']} ms")
                check("返回 record_id", isinstance(d["record_id"], int), str(d["record_id"]))
                check("图片可访问", c.get(base + d["image_url"]).status_code == 200, d["image_url"])
                print(f"      Top-1：{preds[0]['name_cn']}（{preds[0]['name_en']}）"
                      f" {preds[0]['confidence'] * 100:.2f}%  模型={d['model_name']}")

            # 异常输入
            r = c.post(f"{api}/predict", headers=auth,
                       files={"file": ("a.txt", b"hello world", "text/plain")})
            check("非图片扩展名返回 4001", r.status_code == 400 and
                  r.json().get("code") == 4001, f"HTTP {r.status_code} code={r.json().get('code')}")

            r = c.post(f"{api}/predict", headers=auth,
                       files={"file": ("a.jpg", b"\xff\xd8\xff\xe0not-a-real-jpeg", "image/jpeg")})
            check("伪造图片返回 4003", r.json().get("code") == 4003,
                  f"code={r.json().get('code')} msg={r.json().get('message')}")

            big = b"\xff\xd8\xff" + b"0" * (6 * 1024 * 1024)
            r = c.post(f"{api}/predict", headers=auth,
                       files={"file": ("big.jpg", big, "image/jpeg")})
            check("超过 5MB 返回 4002", r.status_code == 413 and
                  r.json().get("code") == 4002, f"HTTP {r.status_code} code={r.json().get('code')}")

            r = c.post(f"{api}/predict", headers=auth,
                       files={"file": (image_path.name, open(image_path, "rb"), "image/jpeg")},
                       data={"model_name": "no_such_model"})
            check("未知模型名返回 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:100]}")

        # ---------------- 6) 花卉百科（M3-5，无需登录）----------------
        print("\n=== 6. 花卉百科（无需登录）===")
        r = c.get(f"{api}/flowers", params={"page": 1, "page_size": 5})
        d = r.json().get("data") or {}
        check("列表返回 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("总数为 102", d.get("total") == 102, f"total={d.get('total')}")
        check("分页生效", len(d.get("items", [])) == 5, f"len={len(d.get('items', []))}")
        check("页数计算正确", d.get("pages") == 21, f"pages={d.get('pages')}")
        if d.get("items"):
            check("含中文名与缩略图", bool(d["items"][0]["name_cn"]),
                  f"{d['items'][0]['name_cn']} / {d['items'][0]['sample_image']}")

        r = c.get(f"{api}/flowers", params={"keyword": "玫瑰"})
        hits = (r.json().get("data") or {}).get("items", [])
        check("搜索“玫瑰”命中", any("玫瑰" in h["name_cn"] for h in hits),
              " / ".join(h["name_cn"] for h in hits[:3]))

        r = c.get(f"{api}/flowers", params={"keyword": "rose"})
        check("英文名搜索命中", (r.json().get("data") or {}).get("total", 0) >= 1,
              f"total={(r.json().get('data') or {}).get('total')}")

        r = c.get(f"{api}/flowers/73")
        fd = r.json().get("data") or {}
        check("详情 class_id=73 为玫瑰", fd.get("name_cn") == "玫瑰" and fd.get("name_en") == "rose",
              f"{fd.get('name_cn')} / {fd.get('name_en')}")
        check("详情含养护字段", bool(fd.get("care_tips") and fd.get("light")), "care_tips/light 非空")

        r = c.get(f"{api}/flowers/999")
        check("越界 class_id 返回 404", r.status_code == 404, f"HTTP {r.status_code}")
        r = c.get(f"{api}/flowers", params={"page": 0})
        check("page=0 返回 422", r.status_code == 422, f"HTTP {r.status_code}")

        # ---------------- 7) 识别历史（M3-6）----------------
        print("\n=== 7. 识别历史（需登录 + 数据隔离）===")
        r = c.get(f"{api}/history")
        check("未登录访问历史返回 401", r.status_code == 401, f"HTTP {r.status_code}")

        r = c.get(f"{api}/history", headers=auth, params={"page": 1, "page_size": 5})
        d = r.json().get("data") or {}
        check("历史列表返回 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("至少有 1 条记录", d.get("total", 0) >= 1, f"total={d.get('total')}")
        own_record_id = d["items"][0]["id"] if d.get("items") else None
        if own_record_id:
            check("列表项含图片与中文名",
                  bool(d["items"][0]["image_url"] and d["items"][0]["name_cn"]),
                  f"{d['items'][0]['name_cn']} {d['items'][0]['image_url']}")

            r = c.get(f"{api}/history/{own_record_id}", headers=auth)
            det = r.json().get("data") or {}
            check("详情含完整 Top-5", len(det.get("top5", [])) == 5, f"len={len(det.get('top5', []))}")

            # 数据隔离：注册第二个用户 B，尝试访问 A 的记录
            sufs = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            ub = f"smoke_b_{sufs}"
            c.post(f"{api}/auth/register",
                   json={"username": ub, "email": f"{ub}@example.com", "password": password})
            tok_b = c.post(f"{api}/auth/login",
                           json={"username": ub, "password": password}).json()["data"]["access_token"]
            auth_b = {"Authorization": f"Bearer {tok_b}"}

            r = c.get(f"{api}/history/{own_record_id}", headers=auth_b)
            check("用户 B 读用户 A 的记录返回 404", r.status_code == 404, f"HTTP {r.status_code}")
            r = c.delete(f"{api}/history/{own_record_id}", headers=auth_b)
            check("用户 B 删用户 A 的记录返回 404", r.status_code == 404, f"HTTP {r.status_code}")

            r = c.get(f"{api}/history", headers=auth_b)
            check("用户 B 的历史为空", (r.json().get("data") or {}).get("total") == 0,
                  f"total={(r.json().get('data') or {}).get('total')}")

            # 删除自己的记录
            r = c.delete(f"{api}/history/{own_record_id}", headers=auth)
            check("删除自己的记录成功", r.status_code == 200, f"HTTP {r.status_code}")
            r = c.get(f"{api}/history/{own_record_id}", headers=auth)
            check("删除后再查返回 404", r.status_code == 404, f"HTTP {r.status_code}")

        # ---------------- 8) 模型对比与统计（M3-7）----------------
        print("\n=== 8. 模型对比与统计 ===")
        r = c.get(f"{api}/models")
        models = r.json().get("data") or []
        check("模型列表返回 4 个", len(models) == 4, f"len={len(models)}")
        check("按 Top-1 降序", all(models[i]["top1"] >= models[i + 1]["top1"]
                                for i in range(len(models) - 1)),
              str([m["top1"] for m in models]))
        check("有且仅有 1 个默认模型", sum(1 for m in models if m["is_default"]) == 1,
              next((m["name"] for m in models if m["is_default"]), None))
        check("指标含预训练来源", all(m["pretrained"] for m in models),
              models[0]["pretrained"][:40] if models else "")

        r = c.get(f"{api}/models/vit_b16/curves")
        pts = (r.json().get("data") or {}).get("points", [])
        check("训练曲线有数据点", len(pts) >= 10, f"points={len(pts)}")
        if pts:
            check("曲线点含 val_top1", pts[0].get("val_top1") is not None, str(pts[0])[:90])

        r = c.get(f"{api}/models/vit_b16/confusion")
        conf = r.json().get("data") or {}
        matrix = conf.get("matrix") or []
        check("混淆矩阵为 102×102", len(matrix) == 102 and len(matrix[0]) == 102,
              f"{len(matrix)}×{len(matrix[0]) if matrix else 0}")
        check("含类别名（102）", len(conf.get("class_names", [])) == 102,
              f"len={len(conf.get('class_names', []))}")
        check("含最易混淆对", len(conf.get("top_pairs", [])) > 0,
              f"top1={conf['top_pairs'][0] if conf.get('top_pairs') else None}")
        check("含混淆矩阵图 URL", bool(conf.get("figure")), conf.get("figure", ""))

        r = c.get(f"{api}/models/nope/curves")
        check("未知模型返回 404", r.status_code == 404, f"HTTP {r.status_code}")

        r = c.get(f"{api}/stats/overview")
        check("未登录访问统计返回 401", r.status_code == 401, f"HTTP {r.status_code}")

        # 先造一条新记录，才能验证聚合逻辑**算得对**（而不是只验证返回了零）
        with open(image_path, "rb") as fh:
            c.post(f"{api}/predict", headers=auth,
                   files={"file": (image_path.name, fh, "image/jpeg")})

        r = c.get(f"{api}/stats/overview", headers=auth)
        st = r.json().get("data") or {}
        check("统计概览返回 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("识别总次数 ≥ 1", st.get("total", 0) >= 1, f"total={st.get('total')}")
        check("今日次数 ≥ 1", st.get("today", 0) >= 1, f"today={st.get('today')}")
        check("平均置信度在合理区间（80—100%）",
              80 <= st.get("avg_confidence", 0) <= 100,
              f"avg_confidence={st.get('avg_confidence')}%")
        check("识别到的不同花卉数 ≥ 1", st.get("unique_classes", 0) >= 1,
              f"unique_classes={st.get('unique_classes')}")
        buckets = st.get("confidence_buckets", [])
        check("置信度分档 5 项", len(buckets) == 5,
              str([(b["label"], b["count"]) for b in buckets]))
        check("分档计数之和 == 总次数（聚合自洽）",
              sum(b["count"] for b in buckets) == st.get("total"),
              f"sum={sum(b['count'] for b in buckets)} vs total={st.get('total')}")
        check("高置信档有计数（模型很自信）",
              sum(b["count"] for b in buckets if b["label"] == "95—100%") >= 1,
              str([(b["label"], b["count"]) for b in buckets]))
        trend = st.get("daily_trend", [])
        check("近 7 天趋势 7 项", len(trend) == 7, f"len={len(trend)}")
        check("今日趋势 > 0", trend and trend[-1]["count"] >= 1,
              f"{trend[-1] if trend else None}")
        check("趋势合计 == 总次数（7 天内）",
              sum(d["count"] for d in trend) == st.get("total"),
              f"sum={sum(d['count'] for d in trend)} vs total={st.get('total')}")
        tops = st.get("top_classes", [])
        check("Top 类别非空且含中文名", bool(tops) and bool(tops[0]["name_cn"]),
              f"{tops[0] if tops else None}")
        check("Top 类别计数 == 总次数（只有一类）",
              sum(t["count"] for t in tops) == st.get("total"),
              f"sum={sum(t['count'] for t in tops)} vs total={st.get('total')}")

        # 图表静态资源
        if conf.get("figure"):
            r = c.get(base + conf["figure"])
            check("混淆矩阵图可访问", r.status_code == 200,
                  f"HTTP {r.status_code} {len(r.content)} bytes")

        r = c.get(f"{base}/static/flowers/073.jpg")
        check("百科缩略图可访问", r.status_code == 200, f"HTTP {r.status_code}")

        r = c.get(f"{base}/docs")
        check("/docs 可访问", r.status_code == 200, f"HTTP {r.status_code}")

        r = c.get(f"{base}/openapi.json")
        paths = sorted((r.json() or {}).get("paths", {}).keys())
        check("OpenAPI 已注册全部接口", len(paths) >= 10, f"{len(paths)} 条路径")

    # ---------------- 汇总 ----------------
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    print("\n" + "=" * 70)
    print(f"冒烟结果：{passed}/{total} 通过")
    if passed < total:
        print("失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

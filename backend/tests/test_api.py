"""后端接口测试（pytest + FastAPI TestClient）。

覆盖 docs/04 M3-8 的要求：认证 3 例、识别 4 例、权限 2 例、百科 2 例，
并补充模型对比与统计的聚合自洽性检查。
"""

from __future__ import annotations

import pytest

# ===========================================================================
# 认证（3 例）
# ===========================================================================


class TestAuth:
    def test_register_success(self, client, api):
        r = client.post(
            f"{api}/auth/register",
            json={"username": "alice_01", "email": "alice01@example.com", "password": "Alice12345"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["username"] == "alice_01"
        assert "password" not in body["data"]  # 绝不回显密码字段
        assert "password_hash" not in body["data"]

    def test_register_duplicate_conflict(self, client, api):
        payload = {"username": "bob_01", "email": "bob01@example.com", "password": "Bob123456"}
        assert client.post(f"{api}/auth/register", json=payload).status_code == 200
        # 用户名重复
        r = client.post(f"{api}/auth/register",
                        json={**payload, "email": "other@example.com"})
        assert r.status_code == 409
        assert "用户名" in r.json()["message"]
        # 邮箱重复
        r = client.post(f"{api}/auth/register",
                        json={**payload, "username": "bob_02"})
        assert r.status_code == 409
        assert "邮箱" in r.json()["message"]

    @pytest.mark.parametrize(
        "payload, reason",
        [
            ({"username": "c_01", "email": "bad-email", "password": "Cpass1234"}, "邮箱格式"),
            ({"username": "c_02", "email": "c02@example.com", "password": "short1"}, "密码过短"),
            ({"username": "c_03", "email": "c03@example.com", "password": "allletters"}, "缺数字"),
            ({"username": "c_04", "email": "c04@example.com", "password": "12345678"}, "缺字母"),
        ],
    )
    def test_register_validation(self, client, api, payload, reason):
        r = client.post(f"{api}/auth/register", json=payload)
        assert r.status_code == 422, f"{reason} 应被拒绝：{r.text}"
        assert r.json()["code"] == 4220


# ===========================================================================
# 识别（4 例）
# ===========================================================================


class TestPredict:
    def test_predict_success(self, client, api, user_token, sample_image_bytes, model_ready):
        if not model_ready:
            pytest.skip("模型未加载（缺少权重或显存不足）")
        headers, _ = user_token
        r = client.post(
            f"{api}/predict",
            headers=headers,
            files={"file": ("flower.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]

        preds = data["predictions"]
        assert len(preds) == 5
        assert all(p["name_cn"] and p["name_en"] for p in preds)
        assert all(0 < p["confidence"] <= 1 for p in preds)
        # 降序
        assert all(preds[i]["confidence"] >= preds[i + 1]["confidence"] for i in range(4))
        # softmax 在 102 类上归一化 → Top-5 之和必然 ≤ 1
        assert sum(p["confidence"] for p in preds) <= 1.0001
        assert 0 <= preds[0]["class_id"] <= 101
        assert data["record_id"] > 0
        assert data["model_name"]

    def test_predict_unsupported_type(self, client, api, user_token):
        headers, _ = user_token
        r = client.post(f"{api}/predict", headers=headers,
                        files={"file": ("note.txt", b"hello", "text/plain")})
        assert r.status_code == 400
        assert r.json()["code"] == 4001
        assert "不支持" in r.json()["message"]

    def test_predict_broken_image(self, client, api, user_token):
        headers, _ = user_token
        r = client.post(f"{api}/predict", headers=headers,
                        files={"file": ("fake.jpg", b"\xff\xd8\xff\xe0not-an-image", "image/jpeg")})
        assert r.json()["code"] == 4003
        assert "损坏" in r.json()["message"]

    def test_predict_too_large(self, client, api, user_token):
        headers, _ = user_token
        big = b"\xff\xd8\xff" + b"0" * (6 * 1024 * 1024)
        r = client.post(f"{api}/predict", headers=headers,
                        files={"file": ("big.jpg", big, "image/jpeg")})
        assert r.status_code == 413
        assert r.json()["code"] == 4002


# ===========================================================================
# 权限（2 例）
# ===========================================================================


class TestPermission:
    def test_predict_requires_login(self, client, api):
        r = client.post(f"{api}/predict")
        assert r.status_code == 401
        assert r.json()["code"] == 4010
        assert "登录" in r.json()["message"]

    def test_history_isolated_between_users(self, client, api, user_token,
                                           sample_image_bytes, model_ready):
        if not model_ready:
            pytest.skip("模型未加载")
        headers_a, _ = user_token
        r = client.post(f"{api}/predict", headers=headers_a,
                        files={"file": ("f.jpg", sample_image_bytes, "image/jpeg")})
        record_id = r.json()["data"]["record_id"]

        # 用户 B
        client.post(f"{api}/auth/register",
                    json={"username": "eve_01", "email": "eve01@example.com",
                          "password": "Eve123456"})
        tok_b = client.post(f"{api}/auth/login",
                            json={"username": "eve_01", "password": "Eve123456"}
                            ).json()["data"]["access_token"]
        headers_b = {"Authorization": f"Bearer {tok_b}"}

        assert client.get(f"{api}/history/{record_id}", headers=headers_b).status_code == 404
        assert client.delete(f"{api}/history/{record_id}", headers=headers_b).status_code == 404
        assert client.get(f"{api}/history", headers=headers_b).json()["data"]["total"] == 0
        # 本人可读
        assert client.get(f"{api}/history/{record_id}", headers=headers_a).status_code == 200


# ===========================================================================
# 百科（2 例）
# ===========================================================================


class TestFlowers:
    def test_list_and_paging(self, client, api):
        r = client.get(f"{api}/flowers", params={"page": 1, "page_size": 10})
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["total"] == 102
        assert len(d["items"]) == 10
        assert d["pages"] == 11
        # 无需登录
        assert client.get(f"{api}/flowers").status_code == 200
        # 边界
        assert client.get(f"{api}/flowers", params={"page": 0}).status_code == 422
        assert client.get(f"{api}/flowers", params={"page_size": 101}).status_code == 422

    def test_detail_and_missing(self, client, api):
        d = client.get(f"{api}/flowers/73").json()["data"]
        assert d["name_cn"] == "玫瑰"
        assert d["name_en"] == "rose"
        assert d["care_tips"] and d["light"]
        assert client.get(f"{api}/flowers/999").status_code == 404
        assert client.get(f"{api}/flowers/-1").status_code == 404


# ===========================================================================
# 模型对比与统计（补充）
# ===========================================================================


class TestModelsAndStats:
    def test_models_sorted_and_single_default(self, client, api):
        models = client.get(f"{api}/models").json()["data"]
        assert len(models) == 4
        assert all(models[i]["top1"] >= models[i + 1]["top1"] for i in range(3))
        assert sum(1 for m in models if m["is_default"]) == 1
        assert models[0]["top1"] >= 95.0, "最优模型应达到验收目标 Top-1 ≥ 95%"

    def test_confusion_matrix_shape(self, client, api):
        d = client.get(f"{api}/models/vit_b16/confusion").json()["data"]
        assert len(d["matrix"]) == 102
        assert all(len(row) == 102 for row in d["matrix"])
        assert len(d["class_names"]) == 102
        assert d["top_pairs"], "应有最易混淆类别对"
        # 关键不变量：对角线之和 == Top-1 命中数。
        # （早期 ml/evaluate.py 原地 fill_diagonal(0)，导致对角线全为 0，
        #   报告里的混淆矩阵图看起来像"没有任何正确预测"，正是这条断言抓出来的。）
        diag = sum(d["matrix"][i][i] for i in range(102))
        expected = round(d["num_samples"] * d["top1"] / 100)
        assert abs(diag - expected) <= 2, (
            f"对角线之和 {diag} 应等于 Top-1 命中数 {expected}"
            f"（Top-1={d['top1']}%, n={d['num_samples']}）"
        )
        # 对角线应严格为正（否则说明矩阵被破坏）
        assert diag > 0

    def test_stats_requires_login_and_aggregates(self, client, api, user_token,
                                                 sample_image_bytes, model_ready):
        assert client.get(f"{api}/stats/overview").status_code == 401
        if not model_ready:
            pytest.skip("模型未加载")
        headers, _ = user_token
        client.post(f"{api}/predict", headers=headers,
                    files={"file": ("f.jpg", sample_image_bytes, "image/jpeg")})
        d = client.get(f"{api}/stats/overview", headers=headers).json()["data"]
        assert d["total"] >= 1
        assert d["today"] >= 1
        assert 0 < d["avg_confidence"] <= 100
        assert len(d["confidence_buckets"]) == 5
        assert len(d["daily_trend"]) == 7
        # 聚合自洽
        assert sum(b["count"] for b in d["confidence_buckets"]) == d["total"]
        assert sum(x["count"] for x in d["daily_trend"]) == d["total"]
        assert sum(t["count"] for t in d["top_classes"]) == d["total"]

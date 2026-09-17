import json
import math
from urllib.request import Request, urlopen

from fly_trader.dashboard import DashboardState, start_dashboard


def test_local_dashboard_serves_state_and_pause_control():
    state = DashboardState()
    state.update({"symbol": "AAPL", "stable_action": "HOLD"}, append=True)
    state.update_account("US", {"status": "ok", "equity": 100000.0})
    server = start_dashboard(state, port=0, open_browser=False)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/state") as response:
            payload = json.load(response)
        assert payload["latest"]["symbol"] == "AAPL"
        assert len(payload["series"]) == 1
        assert payload["accounts"]["US"]["equity"] == 100000.0

        request = Request(base + "/api/pause", method="POST",
                          data=b'{"paused":true}',
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert json.load(response)["paused"] is True
        assert state.is_paused()

        with urlopen(base + "/") as response:
            html = response.read().decode("utf-8")
            assert "果蝇交易实验台" in html
            assert 'data-market="HK"' in html
            assert 'data-market="US"' in html
            assert "持仓与交易统计" in html
            assert "本地模拟成交风控" in html
            assert 'id="riskChecks"' in html
            assert "果蝇原始信号（主指标）" in html
            assert "人工过滤信号（对照）" in html
            assert "风控可执行信号（对照）" in html
            assert "实验资金与收益对比" in html
        with urlopen(base + "/dashboard.js") as response:
            script = response.read().decode("utf-8")
            for label in ("累计买入", "累计卖出", "持仓成本价", "已实现盈亏", "未实现盈亏"):
                assert label in script
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_replaces_non_finite_numbers_with_json_null():
    state = DashboardState()
    state.update_asset("WAITING", {"quote_age_s": math.inf, "raw_score": math.nan})
    snapshot = state.snapshot()
    assert snapshot["assets"]["WAITING"]["quote_age_s"] is None
    assert snapshot["assets"]["WAITING"]["raw_score"] is None
    json.dumps(snapshot, allow_nan=False)

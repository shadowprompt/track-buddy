"""在线地理编码配置的测试。

守住的两个坑（都是实测踩过的）：

1. **`.env` 只在 server.py 里加载** —— CLI、测试脚本、直接 import
   `app.pipeline` 的路径全都读不到 key，看起来就像「key 没配」。
   现在 `app/config.py` 在模块导入时自己加载。

2. **「有 key」不等于「key 能用」** —— 高德 key 绑定平台类型，
   「Web端(JS API)」的 key 调 Web 服务接口会返回
   `USERKEY_PLAT_NOMATCH`（10009）。
   只检查 key 是否存在，会让「配错类型」看起来像「服务正常」。
   现在 `probe_online()` 用真实端点主动打一次并给出修复建议。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def test_dotenv_loaded_on_import():
    """import app.config 就应该能读到 .env 里的变量。"""
    # 用子进程测，避免本进程已 import 过 config 的缓存影响
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from app.config import CONFIG;"
        "print('AMAP' if CONFIG.amap_key else 'NONE')"
        % str(ROOT)
    )
    env = dict(os.environ)
    # 故意清掉环境变量里的 AMAP_KEY，确保只能靠 .env 提供
    env.pop("AMAP_KEY", None)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(ROOT), env=env, timeout=60)
    assert out.returncode == 0, f"子进程失败: {out.stderr[-400:]}"
    got = out.stdout.strip()
    # 若仓库里有 .env 则必须读到；没有 .env 时读不到也算正常
    if (ROOT / ".env").exists():
        assert got == "AMAP", (
            f"import app.config 未自动加载 .env（读到 {got}）")
        print("  [OK] import app.config 即自动加载 .env")
    else:
        print("  [SKIP] 仓库无 .env，跳过")


def test_probe_reports_hint():
    """探测失败时必须给出可读的 hint，而不是只丢一个错误码。"""
    from app.geocode import Geocoder

    src = (ROOT / "app" / "geocode.py").read_text(encoding="utf-8")
    assert "def probe_online" in src, "缺少 probe_online"
    assert "USERKEY_PLAT_NOMATCH" in src, "未处理 10009 平台类型不匹配"
    assert "Web服务" in src, "未给出正确的 key 类型指引"
    assert "_last_error" in src, "未记录失败原因"
    print("  [OK] 探测逻辑含平台类型诊断与修复指引")


def test_probe_is_wired_to_health():
    """health 接口必须调用探测，否则「配错 key」依然看不出来。"""
    src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
    assert "probe_online()" in src, "health 未调用 probe_online"
    assert '"probe"' in src, "health 未返回 probe 字段"
    print("  [OK] health 已接入主动探测")


def test_amap_error_not_silent():
    """_amap 失败时不能静默 return []，必须留下原因。"""
    src = (ROOT / "app" / "geocode.py").read_text(encoding="utf-8")
    # 找到 _amap 方法体，确认里面有 _last_error 赋值
    i = src.find("def _amap(")
    assert i > 0
    body = src[i:i + 3000]
    assert "self._last_error" in body, "_amap 未记录失败原因"
    print("  [OK] _amap 失败会记录原因")


def test_env_file_change_takes_effect_without_restart():
    """改 .env 文件后必须立即生效，不用重启服务。

    实测踩过的坑：key 属性是 `@property`，看起来像「每次实时读取」，
    但 `load_dotenv()` 是在**模块导入时**把 .env 灌进 `os.environ` 的——
    改文件不改进程环境变量，于是新 key 必须重启服务才生效，
    而现象是「明明改了却不生效」，非常难查。

    现在每次取值都重新读 .env 文件（按 mtime 缓存），
    所以改完立刻可用。
    """
    import shutil
    import time

    from app.config import AppConfig

    base = AppConfig()
    if not base.env_path.exists():
        print("  [SKIP] 仓库无 .env，跳过")
        return

    tmp = base.env_path.with_name(".env_hotreload_test")
    shutil.copy(base.env_path, tmp)
    try:
        cfg = AppConfig(env_path=tmp)
        before = cfg.amap_key
        tmp.write_text("AMAP_KEY=deadbeefdeadbeefdeadbeefdeadbeef\n",
                       encoding="utf-8")
        time.sleep(0.05)
        after = cfg.amap_key
        assert before != after, "改 .env 后取值没变化（需重启才生效）"
        assert after.startswith("deadbeef"), f"未读到文件新值：{after}"
        print("  [OK] 改 .env 立即生效（无需重启）")
    finally:
        tmp.unlink(missing_ok=True)


def test_explicit_env_beats_env_file():
    """显式设置的环境变量优先级高于 .env 文件。"""
    import os

    from app.config import AppConfig

    base = AppConfig()
    if not base.env_path.exists():
        print("  [SKIP] 仓库无 .env，跳过")
        return

    sentinel = "explicit1234567890abcdef12345678"
    os.environ["AMAP_KEY"] = sentinel
    try:
        got = AppConfig().amap_key
        assert got == sentinel, f"显式环境变量未优先：{got}"
        print("  [OK] 显式环境变量优先于 .env")
    finally:
        os.environ.pop("AMAP_KEY", None)


def test_repeated_load_dotenv_keeps_file_authority():
    """重复调用 load_dotenv() 不能把文件里的值误判成「显式设置」。

    踩过的坑：第一次调用把文件值灌进 os.environ，第二次调用时
    os.environ 里已经有了这个 key，于是被误判成用户 export 的，
    结果 .env 的改动永远不生效。
    """
    from app.config import load_dotenv, _FILE_SEEDED_VALUES

    load_dotenv()
    load_dotenv()
    seeded = _FILE_SEEDED_VALUES.get("AMAP_KEY")
    if seeded is None:
        print("  [SKIP] 未从 .env 读到 AMAP_KEY，跳过")
        return
    import os
    assert os.environ.get("AMAP_KEY") == seeded, (
        "重复加载后 os.environ 与 seeded 不一致")
    print("  [OK] 重复加载不会改变文件值的归属")


if __name__ == "__main__":
    print("在线地理编码配置测试：")
    test_dotenv_loaded_on_import()
    test_env_file_change_takes_effect_without_restart()
    test_explicit_env_beats_env_file()
    test_repeated_load_dotenv_keeps_file_authority()
    test_probe_reports_hint()
    test_probe_is_wired_to_health()
    test_amap_error_not_silent()
    print("\n全部通过")

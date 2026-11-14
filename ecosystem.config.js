/**
 * pm2 进程配置 —— track-buddy 后端（Python FastAPI）
 *
 * 用法（在服务器的项目根执行）：
 *   pm2 start ecosystem.config.js
 *   pm2 status
 *   pm2 startup systemd   # 按输出提示再执行那条 sudo 命令
 *   pm2 save
 *
 * 路径全部由 __dirname 推导：项目放在 /home/website/track-buddy 或任何
 * 别的目录都不用改本文件。若 venv 不在 <项目根>/.venv，启动时覆盖：
 *   RIDE_PYTHON=/home/website/venv/bin/python pm2 start ecosystem.config.js
 * 端口/监听地址同理：RIDE_HOST / RIDE_PORT（默认 127.0.0.1:8848）。
 *
 * 🔴 三条硬约束（改之前先想清楚）：
 *   1. instances 必须 = 1、exec_mode = fork：data/last_result.json 是全局覆盖写，
 *      多进程会让 A 的核对页看到 B 的轨迹；pm2 cluster 本来也只支持 Node。
 *   2. watch 必须 false：识别会写 data/uploads、data/cache、__pycache__，
 *      开 watch 会陷入「写文件 → 重启 → 再写」的死循环。
 *   3. 不要把 .env 里的密钥搬进下面的 env：config.py 自己热读项目根 .env
 *      （带 mtime+hash 热更新，改密钥无需重启），而这里 env 里的同名变量
 *      优先级更高（显式环境变量 > .env），反而会造成两边不一致。
 */
'use strict';

const path = require('path');

const ROOT = __dirname;
const PYTHON = process.env.RIDE_PYTHON ||
  path.join(ROOT, '.venv', 'bin', 'python');
const HOST = process.env.RIDE_HOST || '127.0.0.1';
const PORT = String(process.env.RIDE_PORT || '8848');

module.exports = {
    apps: [{
        name: 'track-buddy',
        // 用 run.py 而不是直接 uvicorn：它负责把 src/ 注入 sys.path，
        // 并设置静态页端口占位符；绕过它得自己补 PYTHONPATH。
        script: path.join(ROOT, 'run.py'),
        cwd: ROOT,
        // 不写这行时 pm2 会按 .py 猜到系统 python → ModuleNotFoundError: fastapi
        interpreter: PYTHON,
        args: ['--host', HOST, '--port', PORT],

        exec_mode: 'fork',        // 🔴 绝不能 cluster
        instances: 1,             // 🔴 绝不能 > 1
        watch: false,             // 🔴 绝不能开
        autorestart: true,
        max_restarts: 10,
        min_uptime: '10s',
        restart_delay: 3000,
        kill_timeout: 5000,

        env: { PYTHONUNBUFFERED: '1' },   // 密钥别放这里，见文件头说明
        merge_logs: true,
        time: true
        // 日志走 pm2 默认 ~/.pm2/logs/ridewx-api-{out,error}.log，
        // 避免 /var/log 的写权限问题，也不污染仓库目录
    }]
};

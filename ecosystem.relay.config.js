/**
 * pm2 进程配置 —— 高德 WebService 中转服务（独立于主服务）
 *
 * 用法（在服务器的项目根执行）：
 *   pm2 start ecosystem.relay.config.js
 *   pm2 status
 *   pm2 save
 *
 * 前置条件（服务器项目根 .env）：
 *   AMAP_KEY=<高额度key>          # 已有配置，中转直接复用
 *   AMAP_RELAY_TOKEN=<随机串>     # openssl rand -hex 16 生成
 *
 * 监听 0.0.0.0:8849（本地开发机直连的部署形态，安全靠 .env 里的
 * AMAP_RELAY_TOKEN 兜底——未配置时服务拒绝一切转发）。
 * 改走 nginx 反代时收回来：
 *   RIDE_RELAY_HOST=127.0.0.1 pm2 start ecosystem.relay.config.js --update-env
 *
 * 为什么是独立进程而不是挂在 track-buddy 主服务上：
 *   主服务加载 OCR 模型、重启频繁（调参/发版），而本地开发调试
 *   依赖中转的稳定性；轻量服务单独跑互不干扰。
 *
 * 与主服务一样：instances=1、fork 模式、watch 永远关闭。
 */
'use strict';

const path = require('path');

const ROOT = __dirname;
const PYTHON = process.env.RIDE_PYTHON ||
  path.join(ROOT, '.venv', 'bin', 'python');
const HOST = process.env.RIDE_RELAY_HOST || '0.0.0.0';
const PORT = String(process.env.RIDE_RELAY_PORT || '8849');

module.exports = {
    apps: [{
        name: 'amap-relay',
        // run_relay.py 负责 src/ 注入 sys.path，别绕过它
        script: path.join(ROOT, 'run_relay.py'),
        cwd: ROOT,
        interpreter: PYTHON,
        args: ['--host', HOST, '--port', PORT],

        exec_mode: 'fork',
        instances: 1,
        watch: false,
        autorestart: true,
        max_restarts: 10,
        min_uptime: '10s',
        restart_delay: 3000,
        kill_timeout: 5000,

        env: { PYTHONUNBUFFERED: '1' },   // 密钥只放 .env，不放这里
        merge_logs: true,
        time: true
        // 日志走 pm2 默认 ~/.pm2/logs/amap-relay-{out,error}.log
    }]
};

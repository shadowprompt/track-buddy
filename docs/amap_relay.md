# 高德 API 中转服务

## 背景

项目拿到一个使用额度较高的高德 Web 服务 key，但该 key 只对
**服务器出口 IP** 开放（IP 白名单）。本地开发机直接调
`restapi.amap.com` 必然被拒，真实 key 也不应该复制到本地。

方案：在服务器上跑一个轻量中转服务，本地开发时所有高德请求改走中转，
由中转在服务器侧注入 key 后转发给高德。

```
本地代码 ── GET /amap/v3/place/text?...&relay_token=xxx ──▶ 服务器中转
                                                            │ 注入 AMAP_KEY
                                                            ▼
                                                    restapi.amap.com
```

## 组件

| 文件 | 职责 | 运行位置 |
| --- | --- | --- |
| `run_relay.py` | 中转启动入口（uvicorn） | 服务器 |
| `src/app/amap_relay/server.py` | 中转实现：鉴权 + 固定上游转发 + key 注入 | 服务器 |
| `src/app/amap_relay/client.py` | 本地请求改写：`restapi.amap.com` → 中转 | 本地 / 服务器均可 |
| `ecosystem.relay.config.js` | pm2 配置（进程名 `amap-relay`，端口 8849） | 服务器 |

改写点已接入项目内全部 4 处高德调用：地理编码（`maps/geocoding`）、
路名归属（`maps/roadname/amap.py`）、静态地图（`maps/staticmap`）、
路网吸附（`snap.py`）。**未配置中转时行为与原来完全一致**（零回归）。

## 服务器部署

1. 服务器项目根 `.env` 追加：

   ```bash
   AMAP_KEY=<高额度key>          # 若已有则不用改
   AMAP_RELAY_TOKEN=<随机串>     # openssl rand -hex 16 生成
   ```

   ⚠️ 服务器上**不要配 `AMAP_RELAY_URL`**：配了它，服务器自己的
   高德调用也会改走中转（指向自己或别处），轻则绕一圈空耗、
   重则成环。`AMAP_RELAY_URL` / 占位 `AMAP_KEY` 是**本地开发机**
   的配置，两边不要混。

2. 启动（默认监听 `0.0.0.0:8849`，本地直连，无需 nginx）：

   ```bash
   pm2 start ecosystem.relay.config.js
   pm2 save
   ```

   安全靠令牌兜底（`AMAP_RELAY_TOKEN` 未配置时拒绝一切转发）。
   改走 nginx 反代（可选，挂证书/限流）时收回监听：
   `RIDE_RELAY_HOST=127.0.0.1 pm2 start ecosystem.relay.config.js --update-env`

3. （可选）nginx 反代形态，挂证书并加限流：

   ```nginx
   location /amap-relay/ {
       proxy_pass http://127.0.0.1:8849/;
       limit_req zone=amaprelay burst=20 nodelay;
   }
   ```

4. 验证：服务器本机 `curl http://127.0.0.1:8849/healthz`，应返回
   `{"ok":true,"key_configured":true,"token_configured":true}`。

### 🔴 这台服务器放行端口必须做两层（2026-11-18 联调实测）

`1.117.75.135`（OpenCloudOS）的入站流量要过**两道防火墙**，只开第一道
的现象是：tcpdump 能抓到进来的 SYN，但永远等不到回包、外部连接超时——
将来在这台服务器放行**任何新端口**都要两步全做：

```bash
# 第 1 层：腾讯云控制台 → 防火墙页签 → 添加规则（TCP/端口/0.0.0.0.0.0.0.0）
# 第 2 层：服务器 OS 内的 firewalld（OpenCloudOS 默认只开 22/80/443）
firewall-cmd --permanent --add-port=8849/tcp
firewall-cmd --reload          # 热生效，不需要重启
```

验证是否真通：`ss -tlnp | grep <端口>` 确认监听后，从**本地**（注意
先 `--noproxy '*'` 绕过本地代理）curl 一次；抓包定位用
`sudo tcpdump -ni any port <端口>`——「有 In SYN 无回包」= OS 防火墙拦，
「一个包都没有」= 控制台防火墙拦。

## 本地开发配置

（只配在**本地开发机**的 `.env`，服务器上不要照抄这一节）

本地 `.env`：

```bash
AMAP_RELAY_URL=http://<服务器IP>:8849            # 直连形态；走 nginx 则 https://<域名>/amap-relay
AMAP_RELAY_TOKEN=<与服务器一致的随机串>
AMAP_KEY=relay                                   # 占位即可；provider 靠它判断「可用」
```

之后无需改任何代码：项目内所有高德调用自动改走中转。若本地还留有
真实旧 key，中转也会忽略客户端的 `key` 参数、以服务器侧为准。

## 安全设计

- **令牌必填（fail closed）**：`AMAP_RELAY_TOKEN` 未配置时拒绝所有
  转发请求；令牌经 `X-Relay-Token` 头或 `relay_token` query 参数
  传递，`secrets.compare_digest` 常量时间比较。
- **上游固定**：只转发 `https://restapi.amap.com`，路径必须 `v<数字>/`
  开头，无 SSRF 空间。
- **key 不出服务器**：客户端带来的 `key` 一律丢弃；`relay_token`
  也不转发给高德。
- **错误风格**：鉴权/上游错误均按高德响应风格返回
  `{"status":"0","info":"..."}`（HTTP 401/403/502/503），调用方的
  现有错误处理无需适配。

## 排障速查

| 现象 | 原因 |
| --- | --- |
| `info: INVALID_RELAY_TOKEN` | 两端令牌不一致 / 本地没配 `AMAP_RELAY_TOKEN` |
| `info: RELAY_KEY_NOT_CONFIGURED` | 服务器 `.env` 缺 `AMAP_KEY` |
| `info: UPSTREAM_ERROR` | 服务器出网异常或高德超时，看 pm2 日志 |
| 高德返回 IP 白名单错误 | 中转没部署在白名单内的服务器上 |
| `relay` 后仍直连高德 | 本地 `.env` 的 `AMAP_RELAY_URL` 没写或没生效（热读，保存即生效） |

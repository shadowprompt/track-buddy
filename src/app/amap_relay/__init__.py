"""高德 WebService 中转（服务器端转发 + 本地端改写）。

背景
----
项目拿到了一个使用额度较高的高德 Web 服务 key，但该 key 只允许
服务器的出口 IP 调用（IP 白名单）。本地开发机直接请求
``restapi.amap.com`` 必然被拒，本地也没有（不该有）真实 key。

方案：在服务器上跑一个轻量中转服务（``server.py``），本地开发时把
``AMAP_RELAY_URL`` 指向它。本地所有原本发往 ``restapi.amap.com`` 的
请求由 ``client.py`` 改写到中转，中转注入服务器侧 ``AMAP_KEY`` 后
转发给高德。真实 key 永远不出服务器。

两个子模块的职责边界：
  - ``server.py``  只在**服务器**上运行（入口 ``run_relay.py``）；
  - ``client.py``  只在**调用方**进程里做 URL 改写，不发起网络请求。

中转只转发高德 WebService（固定上游 ``restapi.amap.com``，路径必须是
``v<数字>/`` 开头），并以访问令牌鉴权，防止额度被第三方白嫖。
"""

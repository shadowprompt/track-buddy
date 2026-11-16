// 后端服务地址（track-buddy FastAPI）。
//
// 线上（当前默认）：https://track.daozhao.com
// 本地开发者工具联调：把 USE_LOCAL 改成 true，走 http://127.0.0.1:8848
//   （project.config.json 已设 urlCheck=false，等价于「不校验合法域名」）。
// 真机预览连本机时，把 LOCAL_BASE_URL 换成电脑的局域网 IP。
//
// 🔴 正式发布前必须在微信公众平台配置（域名需 https 且已备案）：
//   request 合法域名      -> https://track.daozhao.com 和 https://gateway.daozhao.com
//   downloadFile 合法域名 -> https://track.daozhao.com
//   识别上传是 wx.request + 手工 multipart，没用 wx.uploadFile，
//   所以不需要 uploadFile 合法域名。
//
// 登录（token 签发）收敛到 gateway：POST /wxminTrackBuddy/login，契约与
// track-buddy 的 /api/auth/wxlogin 逐字段一致（{token, expires_in, union_uid}）。
// 🔴 不要改全局 BASE_URL——识别、记录、图片下载仍全部直连 track-buddy；
//    gateway 只负责发 token。USE_LOCAL（本地联调）时 gateway 没在本地跑，
//    LOGIN_BASE_URL 置空 = 登录直接走 BASE_URL 的 /api/auth/wxlogin。
var USE_LOCAL = false;
var LOCAL_BASE_URL = 'http://127.0.0.1:8848';
var ONLINE_BASE_URL = 'https://track.daozhao.com';
var ONLINE_LOGIN_BASE_URL = 'https://gateway.daozhao.com';

module.exports = {
  BASE_URL: USE_LOCAL ? LOCAL_BASE_URL : ONLINE_BASE_URL,
  LOGIN_BASE_URL: USE_LOCAL ? '' : ONLINE_LOGIN_BASE_URL
};

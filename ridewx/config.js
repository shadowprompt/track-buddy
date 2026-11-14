// 后端服务地址（track-buddy FastAPI）。
//
// 线上（当前默认）：https://track.daozhao.com
// 本地开发者工具联调：把 USE_LOCAL 改成 true，走 http://127.0.0.1:8848
//   （project.config.json 已设 urlCheck=false，等价于「不校验合法域名」）。
// 真机预览连本机时，把 LOCAL_BASE_URL 换成电脑的局域网 IP。
//
// 🔴 正式发布前必须在微信公众平台配置（域名需 https 且已备案）：
//   request 合法域名      -> https://track.daozhao.com
//   downloadFile 合法域名 -> https://track.daozhao.com
//   识别上传是 wx.request + 手工 multipart，没用 wx.uploadFile，
//   所以不需要 uploadFile 合法域名。
var USE_LOCAL = false;
var LOCAL_BASE_URL = 'http://127.0.0.1:8848';
var ONLINE_BASE_URL = 'https://track.daozhao.com';

module.exports = {
  BASE_URL: USE_LOCAL ? LOCAL_BASE_URL : ONLINE_BASE_URL
};

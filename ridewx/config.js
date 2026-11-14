// 后端服务地址（meituan-ride FastAPI，默认 8848 端口）。
//
// - 开发者工具联调：保持 127.0.0.1 即可。project.config.json 已设置
//   urlCheck=false（等价于工具里勾选「不校验合法域名」）。
// - 真机预览：手机上打开调试模式后，把这里改成电脑的局域网 IP，
//   例如 http://192.168.1.20:8848（手机和电脑须在同一 Wi-Fi）。
// - 正式发布：需要 https 域名并在微信公众平台配置 request 合法域名。
module.exports = {
  BASE_URL: 'http://127.0.0.1:8848'
};

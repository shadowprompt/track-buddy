App({
  globalData: {
    // 最近一次识别结果（/api/recognize 返回的 payload），
    // index 页写入，result 页读取——数据量含 GPX/TCX 文本，
    // 不适合走页面参数传递。
    lastResult: null
  }
});

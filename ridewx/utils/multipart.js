// 手工构造 multipart/form-data 请求体。
//
// 为什么不用 wx.uploadFile：它一个请求只能携带一个文件，而
// /api/recognize 的契约是一次请求同时传 file（概览图，必填）、
// details（细节图，可多张）、summary（订单详情截图，可选）。
// wx.request 支持 ArrayBuffer 请求体，配合手动拼 multipart 即可。

/** 字符串 → UTF-8 字节。城市名、路名等表单值可能含中文。 */
function utf8(str) {
  var out = [];
  for (var i = 0; i < str.length; ) {
    var code = str.codePointAt(i);
    if (code < 0x80) {
      out.push(code);
    } else if (code < 0x800) {
      out.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      out.push(
        0xe0 | (code >> 12),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f)
      );
    } else {
      out.push(
        0xf0 | (code >> 18),
        0x80 | ((code >> 12) & 0x3f),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f)
      );
    }
    i += code > 0xffff ? 2 : 1;
  }
  return new Uint8Array(out);
}

function concat(parts) {
  var total = 0;
  for (var i = 0; i < parts.length; i++) total += parts[i].length;
  var buf = new Uint8Array(total);
  var off = 0;
  for (var j = 0; j < parts.length; j++) {
    buf.set(parts[j], off);
    off += parts[j].length;
  }
  return buf.buffer;
}

/**
 * @param {Array<{name:string, value:string}>} fields 普通表单字段
 * @param {Array<{name:string, filename:string, contentType:string, data:ArrayBuffer}>} files 文件字段
 * @returns {{contentType:string, buffer:ArrayBuffer}}
 */
function build(fields, files) {
  var boundary = '----rideboundary' + Date.now();
  var parts = [];
  var i;
  for (i = 0; i < (fields || []).length; i++) {
    var f = fields[i];
    parts.push(
      utf8(
        '--' + boundary +
        '\r\nContent-Disposition: form-data; name="' + f.name + '"' +
        '\r\n\r\n' + f.value + '\r\n'
      )
    );
  }
  for (i = 0; i < (files || []).length; i++) {
    var file = files[i];
    parts.push(
      utf8(
        '--' + boundary +
        '\r\nContent-Disposition: form-data; name="' + file.name +
        '"; filename="' + file.filename + '"' +
        '\r\nContent-Type: ' + (file.contentType || 'application/octet-stream') +
        '\r\n\r\n'
      )
    );
    parts.push(new Uint8Array(file.data));
    parts.push(utf8('\r\n'));
  }
  parts.push(utf8('--' + boundary + '--\r\n'));
  return {
    contentType: 'multipart/form-data; boundary=' + boundary,
    buffer: concat(parts)
  };
}

module.exports = { build: build, utf8: utf8 };

// Run docs/setvision.js on a raw RGBA file: node tests/run_js.js <file> <width> <height>
// Prints the detected cards as JSON. Used by tests/js_parity.py.
const fs = require("fs");
const path = require("path");
const SetVision = require(path.join(__dirname, "..", "docs", "setvision.js"));
const [file, w, h] = process.argv.slice(2);
const buf = fs.readFileSync(file);
const img = { data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.length), width: +w, height: +h };
const t0 = Date.now();
const cards = SetVision.detectCards(img);
console.log(JSON.stringify({ ms: Date.now() - t0, cards }));

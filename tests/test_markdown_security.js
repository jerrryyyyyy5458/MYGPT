const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = fs.readFileSync(
  path.join(__dirname, "..", "static", "index.html"),
  "utf8"
);
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const start = script.indexOf("function escapeHtml");
const end = script.indexOf("/* ---------------- Messages", start);
assert.notEqual(start, -1, "Markdown renderer should be present");
assert.notEqual(end, -1, "Markdown renderer boundary should be present");
assert.doesNotThrow(() => new vm.Script(script), "inline application JavaScript should parse");

const markdown = { URL };
vm.runInNewContext(script.slice(start, end), markdown);

test("normal HTTP and HTTPS Markdown links remain usable", () => {
  const output = markdown.inlineMd(
    "[secure](https://example.test/docs?a=1&b=2) [plain](http://example.test)"
  );
  assert.match(output, /href="https:\/\/example\.test\/docs\?a=1&amp;b=2"/);
  assert.match(output, /href="http:\/\/example\.test\//);
  assert.match(output, /rel="noopener noreferrer"/);
});

test("quotes in URLs cannot inject HTML attributes", () => {
  const output = markdown.inlineMd(
    '[click](https://safe.test/"onmouseover="alert(1))'
  );
  assert.match(output, /<a href=/);
  assert.doesNotMatch(output, /\sonmouseover\s*=/i);
  assert.doesNotMatch(output, /javascript:/i);
});

test("unsafe URL protocols are rendered as text, not links", () => {
  const output = markdown.inlineMd(
    "[run](javascript:alert(1)) [data](data:text/html,alert)"
  );
  assert.doesNotMatch(output, /<a\s/i);
  assert.match(output, /javascript:/i);
  assert.match(output, /data:text\/html/i);
});

test("link labels cannot inject HTML", () => {
  const output = markdown.inlineMd(
    "[<img src=x onerror=alert(1)>](https://safe.test/)"
  );
  assert.match(output, /&lt;img/);
  assert.doesNotMatch(output, /<img/i);
  assert.doesNotMatch(output, /<[^>]*onerror=/i);
});

test("bold text, inline code, and fenced code keep their formatting", () => {
  assert.match(markdown.inlineMd("**bold** and `code`"), /<strong>bold<\/strong>/);
  assert.match(markdown.inlineMd("**bold** and `code`"), /<code class="inline">code<\/code>/);
  const output = markdown.renderMarkdown("```html\n<img onerror=alert(1)>\n```\n\n**bold**");
  assert.match(output, /<pre class="code"><code>/);
  assert.match(output, /&lt;img onerror=alert\(1\)&gt;/);
  assert.match(output, /<strong>bold<\/strong>/);
});

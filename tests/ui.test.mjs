// 合成数据交互测试：本地 mock 管理接口 + 缓存 Chromium（不联网、不装包）
// 运行：node tests/ui.test.mjs
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PANEL = path.resolve(HERE, '..', 'deploy', 'panel.html');
const require = createRequire(import.meta.url);

function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_CORE,
    'playwright-core'
  ].filter(Boolean);
  for (const c of candidates) {
    try { return require(c); } catch (e) { /* 继续找 */ }
  }
  throw new Error('找不到 playwright-core，可用 PLAYWRIGHT_CORE 指定');
}

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  loadPlaywright().chromium.executablePath(),
  '/usr/bin/google-chrome',
  '/usr/bin/chromium'
].filter(Boolean);

/* ---------------- 合成模型数据 ---------------- */

const BASE_MODELS = [
  { channel: 'openai-embeddings', name: 'text-embedding-3-small', alias: 'kb-openai', automatic: true, mode: 'auto', enabled: true, protocol: 'openai' },
  { channel: 'openai-embeddings', name: 'text-embedding-3-large', alias: '', automatic: false, mode: 'allow', enabled: true, protocol: 'openai' },
  { channel: 'gemini', name: 'gemini-embedding-2', alias: '', automatic: true, mode: 'deny', enabled: false, protocol: 'gemini' },
  { channel: 'cohere-embeddings', name: 'embed-v4.0', alias: 'kb-cohere', automatic: false, mode: 'auto', enabled: false, protocol: 'cohere' },
  { channel: 'gemini', name: 'text-embedding-004', alias: '', automatic: true, mode: 'auto', enabled: true, protocol: 'gemini' }
];

function clone(models) { return models.map((m) => ({ ...m })); }
function effective(mode, automatic) {
  if (mode === 'allow') return true;
  if (mode === 'deny') return false;
  return automatic === true;
}

/* ---------------- mock 服务 ---------------- */

function createMock(html) {
  const state = {
    models: clone(BASE_MODELS),
    revision: 'r1',
    key: 'test-key',
    revSeq: 1,
    requests: [],
    getDelays: [],
    postDelay: 0,
    errorOnce: 0,
    conflictOnce: false
  };

  function snapshot() {
    return { models: clone(state.models), revision: state.revision };
  }

  function reset(models, revision) {
    state.models = clone(models || BASE_MODELS);
    state.revision = revision || 'r1';
    state.revSeq = Number(String(state.revision).replace(/\D/g, '')) || 1;
    state.requests = [];
    state.getDelays = [];
    state.postDelay = 0;
    state.errorOnce = 0;
    state.conflictOnce = false;
  }

  function send(res, status, payload) {
    if (res.writableEnded || res.destroyed) return;
    const body = JSON.stringify(payload);
    try {
      res.writeHead(status, {
        'content-type': 'application/json; charset=utf-8',
        'cache-control': 'no-store',
        'content-length': Buffer.byteLength(body)
      });
      res.end(body);
    } catch (e) { /* 客户端已中止 */ }
  }

  function record(req, body) {
    state.requests.push({
      method: req.method,
      url: req.url,
      headers: { ...req.headers },
      body: body || null
    });
  }

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://mock.local');
    if (req.method === 'GET' && (url.pathname === '/deploy/panel.html' ||
        url.pathname === '/v0/resource/plugins/cpa-embeddings/panel')) {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(html);
      return;
    }
    if (url.pathname !== '/embedding-models/api' &&
        url.pathname !== '/v0/management/cpa-embeddings/models') {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('not found');
      return;
    }
    const token = String(req.headers.authorization || '').replace(/^Bearer\s+/, '');
    if (req.method === 'GET') {
      const delay = state.getDelays.length ? state.getDelays.shift() : 0;
      record(req);
      if (delay) await new Promise((r) => setTimeout(r, delay));
      if (token !== state.key) return send(res, 401, { error: 'unauthorized' });
      return send(res, 200, snapshot());
    }
    if (req.method === 'POST') {
      let raw = '';
      for await (const chunk of req) raw += chunk;
      let body = null;
      try { body = JSON.parse(raw); } catch (e) { body = null; }
      record(req, body);
      if (state.postDelay) await new Promise((r) => setTimeout(r, state.postDelay));
      if (token !== state.key) return send(res, 401, { error: 'unauthorized' });
      if (!body || typeof body !== 'object' || !body.channel || !body.name || !body.mode || !body.revision) {
        return send(res, 400, { error: 'bad request' });
      }
      if (state.errorOnce) {
        const status = state.errorOnce;
        state.errorOnce = 0;
        return send(res, status, { error: 'injected failure' });
      }
      if (state.conflictOnce) {
        // 模拟另一位管理员期间改了别的模型：版本号前进，本条保持不变。
        state.conflictOnce = false;
        state.revision = 'r' + (++state.revSeq);
        const other = state.models.find((m) => m.name === 'text-embedding-004' && m.channel === 'gemini');
        if (other) { other.mode = 'deny'; other.enabled = false; }
        return send(res, 409, { error: 'revision conflict' });
      }
      if (body.revision !== state.revision) return send(res, 409, { error: 'revision conflict' });
      const target = state.models.find((m) => m.channel === body.channel && m.name === body.name);
      if (!target) return send(res, 404, { error: 'model not found' });
      target.mode = body.mode;
      target.enabled = effective(body.mode, target.automatic);
      state.revision = 'r' + (++state.revSeq);
      return send(res, 200, snapshot());
    }
    res.writeHead(405, { 'content-type': 'text/plain' });
    res.end('method not allowed');
  });

  return { server, state, reset, snapshot };
}

/* ---------------- 断言与页面助手 ---------------- */

const results = [];
async function test(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log('  ok  ' + name);
  } catch (err) {
    results.push({ name, ok: false, err });
    console.log('  FAIL ' + name);
    console.log('       ' + String(err && err.message || err).split('\n').join('\n       '));
  }
}

function rowFor(page, name) {
  return page.locator('#rows .row').filter({ has: page.locator('.model-name', { hasText: name }) });
}

async function connect(page, key = 'test-key') {
  await page.fill('#adminKey', key);
  await page.click('#connectBtn');
  await page.waitForSelector('#rows .row', { timeout: 5000 });
}

async function storageCheck(page) {
  const probe = await page.evaluate(() => ({
    local: localStorage.length,
    session: sessionStorage.length,
    cookie: document.cookie,
    search: location.search,
    hash: location.hash
  }));
  assert.equal(probe.local, 0, 'localStorage 应保持为空');
  assert.equal(probe.session, 0, 'sessionStorage 应保持为空');
  assert.equal(probe.cookie, '', '不应写入 Cookie');
  assert.equal(probe.search, '', '密钥等不应进入查询串');
  assert.equal(probe.hash, '', '不应写入地址栏片段');
}

/* ---------------- 主流程 ---------------- */

const { chromium } = loadPlaywright();
const html = await readFile(PANEL, 'utf8');
const mock = createMock(html);

await new Promise((resolve) => mock.server.listen(0, '127.0.0.1', resolve));
const port = mock.server.address().port;
const base = 'http://127.0.0.1:' + port;

let browser = null;
try {
  const launchOpts = { headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] };
  for (const candidate of CHROME_CANDIDATES) {
    try {
      browser = await chromium.launch({ ...launchOpts, executablePath: candidate });
      break;
    } catch (e) { browser = null; }
  }
  if (!browser) browser = await chromium.launch(launchOpts);
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();

  /* 1. 密钥门、静态资源与浏览器存储 */
  await test('未连接时只显示密钥门，不加载任何外部资源', async () => {
    mock.reset(BASE_MODELS, 'r1');
    await page.goto(base + '/deploy/panel.html');
    assert.equal(await page.locator('#gate').isVisible(), true);
    assert.equal(await page.locator('#console').isVisible(), false);
    assert.equal(await page.locator('#rows .row').count(), 0);
    const probe = await page.evaluate(() => ({
      scripts: [...document.querySelectorAll('script[src]')].length,
      links: [...document.querySelectorAll('link[rel="stylesheet"], link[href]')].length,
      imgs: [...document.querySelectorAll('img')].length,
      resources: performance.getEntriesByType('resource').map((e) => e.name)
    }));
    assert.equal(probe.scripts, 0);
    assert.equal(probe.links, 0);
    assert.equal(probe.imgs, 0);
    for (const r of probe.resources) assert.ok(r.startsWith(base), '资源必须同源：' + r);
    await storageCheck(page);
    assert.equal(mock.state.requests.length, 0, '未提交前不应请求接口');
  });

  /* 2. 连接：鉴权头、密钥清空、统计与区分 */
  await test('连接只携带 Bearer 管理密钥，渲染统计并区分自动识别与最终可用', async () => {
    await connect(page);
    assert.equal(await page.inputValue('#adminKey'), '', '提交后输入框应清空');
    assert.equal(await page.locator('#linkText').textContent(), '已连接 · 5 个模型');
    assert.equal(await page.locator('#sTotal').textContent(), '5');
    assert.equal(await page.locator('#sAuto').textContent(), '3');
    assert.equal(await page.locator('#sOpen').textContent(), '3');
    assert.equal(await page.locator('#revision').textContent(), 'r1');
    assert.equal(await page.locator('#endpoint').textContent(), '/embedding-models/api');

    const get = mock.state.requests.filter((r) => r.method === 'GET');
    assert.equal(get.length, 1);
    assert.equal(get[0].url, '/embedding-models/api');
    assert.equal(get[0].headers.authorization, 'Bearer test-key');
    assert.equal(get[0].headers.cookie, undefined);
    assert.equal(get[0].headers.referer, undefined);
    assert.equal(Object.keys(get[0].headers).filter((h) => h.startsWith('x-')).length, 0);

    const deniedAuto = rowFor(page, 'gemini-embedding-2');
    assert.equal((await deniedAuto.locator('.badge').textContent()).trim(), '常见模型');
    assert.equal((await deniedAuto.locator('.pill').textContent()).trim(), '不对外');
    const manualAllow = rowFor(page, 'text-embedding-3-large');
    assert.equal((await manualAllow.locator('.badge').textContent()).trim(), '未识别');
    assert.equal((await manualAllow.locator('.pill').textContent()).trim(), '对外可用');

    assert.equal(await page.locator('#rows .row button').evaluateAll((bs) => bs.every((b) => b.disabled)), true);
    await storageCheck(page);
  });

  /* 3. 过滤 */
  await test('渠道、关键词与「只看已开放」过滤可用', async () => {
    const options = await page.locator('#channelFilter option').allTextContents();
    assert.deepEqual(options, ['全部渠道', 'cohere-embeddings', 'gemini', 'openai-embeddings']);
    await page.selectOption('#channelFilter', 'gemini');
    assert.equal(await page.locator('#rows .row').count(), 2);
    await page.selectOption('#channelFilter', 'all');
    await page.fill('#search', 'kb-cohere');
    assert.equal(await page.locator('#rows .row').count(), 1);
    assert.equal(await page.locator('#rows .row .model-name').textContent(), 'embed-v4.0');
    await page.fill('#search', '');
    await page.check('#onlyOpen');
    assert.equal(await page.locator('#rows .row').count(), 3);
    assert.equal(await page.locator('#countText').textContent(), '显示 3 / 5 条');
    await page.uncheck('#onlyOpen');
    assert.equal(await page.locator('#rows .row').count(), 5);
  });

  /* 4. 三态下拉 + 逐条保存 */
  await test('三态下拉只激活本行，保存提交完整字段并采纳新快照', async () => {
    const row = rowFor(page, 'embed-v4.0');
    const select = row.locator('select');
    assert.deepEqual(await select.locator('option').evaluateAll((os) => os.map((o) => o.value)), ['auto', 'allow', 'deny']);
    await select.selectOption('allow');
    assert.equal(await row.locator('.preview').textContent(), '保存后：对外可用');
    assert.equal(await row.locator('button').isDisabled(), false);
    const others = page.locator('#rows .row').filter({ hasNot: page.locator('.model-name', { hasText: 'embed-v4.0' }) });
    assert.equal(await others.locator('button').evaluateAll((bs) => bs.every((b) => b.disabled)), true);

    await row.locator('button').click();
    await page.waitForFunction(() => document.getElementById('revision').textContent === 'r2');
    const post = mock.state.requests.filter((r) => r.method === 'POST');
    assert.equal(post.length, 1);
    assert.deepEqual(post[0].body, { channel: 'cohere-embeddings', name: 'embed-v4.0', mode: 'allow', revision: 'r1' });
    assert.equal(post[0].headers.authorization, 'Bearer test-key');
    assert.equal(post[0].headers.cookie, undefined);
    assert.equal(post[0].headers['content-type'], 'application/json');
    assert.equal(await row.locator('select').inputValue(), 'allow');
    assert.equal((await row.locator('.pill').textContent()).trim(), '对外可用');
    assert.equal(await row.locator('button').isDisabled(), true);
    assert.equal(await page.locator('#sOpen').textContent(), '4');
    await storageCheck(page);
  });

  /* 5. 409 冲突 */
  await test('保存冲突显示提示、主动重读、不假装已保存', async () => {
    mock.state.conflictOnce = true;
    await page.waitForFunction(() => document.getElementById('toast').hidden === true, null, { timeout: 4000 });
    const row = rowFor(page, 'gemini-embedding-2');
    const before = mock.state.requests.filter((r) => r.method === 'GET').length;
    await row.locator('select').selectOption('allow');
    await row.locator('button').click();
    await page.waitForFunction(() => document.getElementById('revision').textContent === 'r3');
    assert.ok((await page.locator('#noticeText').textContent()).includes('保存冲突'));
    assert.equal(await page.locator('#notice').isVisible(), true);
    const gets = mock.state.requests.filter((r) => r.method === 'GET');
    assert.equal(gets.length, before + 1, '冲突后应主动重新读取一次');
    assert.equal(await page.locator('#revision').textContent(), 'r3');
    assert.equal((await row.locator('.row-status').textContent()).trim(), '配置已被他人修改，已重新读取');
    assert.equal((await row.locator('.pill').textContent()).trim(), '不对外');
    assert.equal(await row.locator('select').inputValue(), 'allow', '草稿应保留供管理员核对');
    assert.equal(await row.locator('[data-flash]').count(), 0);
    const saved = mock.state.requests.filter((r) => r.method === 'POST');
    assert.equal(saved.length, 2);
    const toastVisible = await page.locator('#toast').isVisible();
    assert.equal(toastVisible, false, '冲突不应出现成功提示');
  });

  /* 6. 其他错误不假装保存 */
  await test('服务 500 时提示失败并保留草稿、版本不变', async () => {
    mock.state.errorOnce = 500;
    const row = rowFor(page, 'text-embedding-3-small');
    const revBefore = await page.locator('#revision').textContent();
    await row.locator('select').selectOption('deny');
    await row.locator('button').click();
    await page.waitForFunction(() => {
      const el = document.querySelector('#notice');
      return el && !el.hidden && document.getElementById('noticeText').textContent.includes('500');
    });
    assert.ok((await page.locator('#noticeText').textContent()).includes('没有生效'));
    assert.equal((await row.locator('.row-status').textContent()).trim(), '保存失败（HTTP 500），未生效');
    assert.equal((await row.locator('.pill').textContent()).trim(), '对外可用', '服务端状态未变，不能显示已保存');
    assert.equal(await page.locator('#revision').textContent(), revBefore);
    assert.equal(await row.locator('select').inputValue(), 'deny');
  });

  /* 7. 重复提交 */
  await test('保存中重复点击只发出一次请求', async () => {
    mock.state.postDelay = 450;
    mock.state.requests = mock.state.requests.filter((r) => r.method === 'GET');
    const row = rowFor(page, 'text-embedding-3-large');
    await row.locator('select').selectOption('auto');
    await page.evaluate(() => {
      const rows = [...document.querySelectorAll('#rows .row')];
      const target = rows.find((r) => r.querySelector('.model-name').textContent === 'text-embedding-3-large');
      const btn = target.querySelector('button');
      btn.click();
      btn.click();
      btn.click();
    });
    assert.equal(await row.locator('button').isDisabled(), true, '提交中按钮应禁用');
    assert.equal((await row.locator('button').textContent()).trim(), '保存中');
    await page.waitForFunction(() => document.getElementById('revision').textContent === 'r4');
    const posts = mock.state.requests.filter((r) => r.method === 'POST');
    assert.equal(posts.length, 1, '重复点击不应重复提交');
    assert.equal((await row.locator('button').textContent()).trim(), '保存');
    mock.state.postDelay = 0;
  });

  /* 8. 过期响应 */
  await test('连续重新读取时，被取代的旧响应不覆盖界面', async () => {
    mock.reset(BASE_MODELS, 'r9');
    mock.state.models = clone(BASE_MODELS).slice(0, 2);
    mock.state.getDelays = [700, 0];
    await page.click('#refreshBtn');
    await page.waitForTimeout(120);
    await page.click('#refreshBtn');
    await page.waitForFunction(() => document.getElementById('revision').textContent === 'r9');
    await page.waitForTimeout(900);
    assert.equal(await page.locator('#revision').textContent(), 'r9');
    assert.equal(await page.locator('#rows .row').count(), 2, '只应显示最新一次读取的结果');
    assert.equal(await page.locator('#notice').isVisible(), false, '中止的旧请求不应报错');
    assert.equal(await page.locator('#rows .row').first().locator('select').isDisabled(), false);
  });

  /* 9. 未加载保护 */
  await test('读取未完成时禁止筛选与保存', async () => {
    mock.reset(BASE_MODELS, 'r1');
    mock.state.getDelays = [800];
    await page.goto(base + '/deploy/panel.html');
    await page.fill('#adminKey', 'test-key');
    await page.click('#connectBtn');
    await page.waitForTimeout(250);
    assert.equal(await page.locator('#board').getAttribute('aria-busy'), 'true');
    assert.equal(await page.locator('#rows .row').count(), 0);
    assert.equal(await page.locator('#channelFilter').isDisabled(), true);
    assert.equal(await page.locator('#search').isDisabled(), true);
    assert.equal(await page.locator('#onlyOpen').isDisabled(), true);
    assert.equal((await page.locator('#refreshBtn').textContent()).trim(), '读取中…');
    await page.waitForSelector('#rows .row', { timeout: 5000 });
    assert.equal(await page.locator('#board').getAttribute('aria-busy'), 'false');
    assert.equal(await page.locator('#channelFilter').isDisabled(), false);
  });

  /* 10. 鉴权失败 */
  await test('密钥无效时退回密钥门并清空输入', async () => {
    await page.goto(base + '/deploy/panel.html');
    await page.fill('#adminKey', 'bad-key');
    await page.click('#connectBtn');
    await page.waitForFunction(() => document.getElementById('gateError').textContent.length > 0);
    assert.ok((await page.locator('#gateError').textContent()).includes('无效或已过期'));
    assert.equal(await page.locator('#console').isVisible(), false);
    assert.equal(await page.locator('#rows .row').count(), 0);
    assert.equal(await page.inputValue('#adminKey'), '');
    await storageCheck(page);
  });

  /* 11. 退出清空 */
  await test('退出后清空内存密钥与数据，且不再发请求', async () => {
    mock.reset(BASE_MODELS, 'r1');
    await page.goto(base + '/deploy/panel.html');
    await connect(page);
    const before = mock.state.requests.length;
    await page.click('#logoutBtn');
    assert.equal(await page.locator('#gate').isVisible(), true);
    assert.equal(await page.locator('#rows .row').count(), 0);
    assert.equal(await page.locator('#keyState').textContent(), '未持有');
    await page.waitForTimeout(250);
    assert.equal(mock.state.requests.length, before, '退出不应产生网络请求');
    await storageCheck(page);
    const probe = await page.evaluate(() => document.documentElement.outerHTML.includes('test-key'));
    assert.equal(probe, false, '密钥不应残留在页面文本中');
  });

  /* 12. 安全文本渲染 */
  await test('模型名/别名中的 HTML 只按文本显示', async () => {
    const evil = [
      { channel: '<b>chan</b>', name: '<img src=x onerror="window.__xss=1">', alias: '<script>window.__xss=2<\/script>', automatic: true, mode: 'auto', enabled: true, protocol: 'openai' }
    ];
    mock.reset(evil, 'rx');
    await page.goto(base + '/deploy/panel.html');
    await connect(page);
    assert.equal(await page.locator('#rows img').count(), 0);
    assert.equal(await page.locator('#rows script').count(), 0);
    assert.equal(await page.locator('#rows b').count(), 0);
    assert.equal(await page.evaluate(() => window.__xss), undefined);
    const nameText = await page.locator('#rows .model-name').textContent();
    assert.ok(nameText.includes('<img src=x onerror='), '应按字面文本显示');
    const aliasText = await page.locator('#rows .model-alias').textContent();
    assert.ok(aliasText.includes('<script>'), '别名应按字面文本显示');
  });

  /* 13. CPA 原生插件路径 */
  await test('在 CPA 插件页内打开时改用同源管理路径', async () => {
    mock.reset(BASE_MODELS, 'r1');
    await page.goto(base + '/v0/resource/plugins/cpa-embeddings/panel');
    assert.equal(await page.locator('#endpoint').textContent(), '/v0/management/cpa-embeddings/models');
    await connect(page);
    const gets = mock.state.requests.filter((r) => r.method === 'GET');
    assert.equal(gets.length, 1);
    assert.equal(gets[0].url, '/v0/management/cpa-embeddings/models');
    assert.equal(await page.locator('#revision').textContent(), 'r1');
    assert.equal(await page.locator('#rows .row').count(), 5);
  });

  /* 14. 静态约束 */
  await test('源码不写浏览器存储、不使用内联 HTML 注入与外部资源', async () => {
    // 注释里可以提到被禁止的 API 名称，静态检查只看实际代码与标记。
    const source = html.replace(/<!--[\s\S]*?-->/g, '');
    for (const banned of ['localStorage', 'sessionStorage', 'document.cookie', 'indexedDB', 'innerHTML', 'outerHTML', 'insertAdjacentHTML', 'document.write']) {
      assert.equal(source.includes(banned), false, '不应出现 ' + banned);
    }
    assert.equal(/<script\s+src=|<link\s+[^>]*href=|<img\s+src=/i.test(source), false, '不应引入外部资源');
    assert.equal(/https?:\/\//i.test(source), false, '不应出现外部 URL');
    assert.ok(source.includes("credentials: 'omit'"), '应使用 credentials: omit');
    assert.ok(source.includes("redirect: 'error'"), '应使用 redirect: error');
    assert.ok(source.includes('referrerPolicy'), '应限制 referrer');
  });
} finally {
  if (browser) await browser.close().catch(() => {});
  await new Promise((r) => mock.server.close(r));
}

const failed = results.filter((r) => !r.ok);
console.log('\n' + (results.length - failed.length) + '/' + results.length + ' 通过');
if (failed.length) {
  for (const f of failed) console.error('失败：' + f.name + '\n' + (f.err && f.err.stack || f.err));
  process.exitCode = 1;
}

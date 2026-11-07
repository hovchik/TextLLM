// node print.js in.html out.pdf: prints the HTML to an A4 PDF with page numbers (Chromium via Playwright).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(inp), { waitUntil: 'networkidle' });
  await p.pdf({ path: out, format: 'A4', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;color:#888;width:100%;text-align:center;font-family:Arial">Text LLM Studio · User manual · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '18mm', bottom: '20mm', left: '16mm', right: '16mm' } });
  await b.close();
})();

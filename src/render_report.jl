# Render the markdown report into a self-contained HTML via marked.js +
# MathJax. Open the result in a browser and Print -> Save as PDF.

const REPORT_MD   = joinpath(@__DIR__, "..", "report", "report.md")
const REPORT_HTML = joinpath(@__DIR__, "..", "report", "report.html")

md = read(REPORT_MD, String)
md_escaped = replace(md,
    "\\" => "\\\\",
    "`"  => "\\`",
    "\${" => "\\\${",
)

html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GEO 384H Final, Recovering the ENSO Attractor</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=JetBrains+Mono&display=swap');
html { -webkit-print-color-adjust: exact; }
body  { font-family: 'Source Serif 4', Georgia, serif;
        max-width: 760px; margin: 2em auto; padding: 0 1.5em;
        color: #222; line-height: 1.55; font-size: 16px; }
h1, h2, h3, h4 { font-weight: 700; line-height: 1.2; color: #111; }
h1 { font-size: 2em;    border-bottom: 2px solid #222; padding-bottom: 6px; }
h2 { font-size: 1.45em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 1.18em; margin-top: 1.5em; }
em { color: #444; }
code { font-family: 'JetBrains Mono', monospace; font-size: 0.9em;
       background: #f4f4f5; padding: 1px 5px; border-radius: 3px; }
pre  { background: #f4f4f5; padding: 0.8em; border-radius: 5px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { padding: 6px 12px; border-bottom: 1px solid #ddd; text-align: left; }
th { background: #fafafa; }
img { max-width: 100%; height: auto; display: block;
      margin: 1em auto; border: 1px solid #eaeaea; border-radius: 5px; }
figure { text-align: center; }
figcaption { font-size: 0.92em; color: #555; }
blockquote { border-left: 3px solid #888; padding-left: 1em; color: #555; }
@media print {
  body { max-width: none; margin: 0; padding: 0.5in; font-size: 11pt; }
  h1, h2, h3 { break-after: avoid; }
  img, table, pre { break-inside: avoid; }
}
#content { display: none; }
</style>
</head>
<body>
<div id="content"></div>

<script>
window.MathJax = {
  tex: {
    inlineMath: [['\$', '\$'], ['\\\\(', '\\\\)']],
    displayMath: [['\$\$', '\$\$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script', 'style'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<script>
const md = `$md_escaped`;
marked.use({ gfm: true, breaks: false });
const target = document.getElementById('content');
target.innerHTML = marked.parse(md);
target.style.display = 'block';
// Re-typeset math after markdown insertion
if (window.MathJax && MathJax.typesetPromise) {
  MathJax.typesetPromise([target]);
} else {
  document.addEventListener('DOMContentLoaded', () => {
    if (window.MathJax) MathJax.typesetPromise([target]);
  });
}
</script>
</body>
</html>
"""

write(REPORT_HTML, html)
println("Wrote $REPORT_HTML")
println("Open it in a browser and use Print → Save as PDF.")

// Copy button for the Drive sharing notice (AICP-274). Delegated, so it works
// for notices inside HTMX-swapped fragments and needs no inline script/handler.
document.addEventListener('click', function (ev) {
  var btn = ev.target && ev.target.closest ? ev.target.closest('[data-copy-text]') : null;
  if (!btn) return;
  var text = btn.getAttribute('data-copy-text') || '';
  if (!text) return;
  function done() {
    var label = btn.getAttribute('data-label') || btn.textContent;
    btn.setAttribute('data-label', label);
    btn.textContent = 'Copied';
    setTimeout(function () { btn.textContent = label; }, 1500);
  }
  function fallback() {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { if (document.execCommand('copy')) done(); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, fallback);
  } else {
    fallback();
  }
});

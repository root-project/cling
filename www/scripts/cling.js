requirejs(["//cdn.jsdelivr.net/highlight.js/9.2.0/highlight.min.js"], function() {
  hljs.initHighlightingOnLoad();
});
function selectMenu(tab) {
  document.getElementById('iheader').contentDocument.getElementById('tab_' + tab).className='selected'
}

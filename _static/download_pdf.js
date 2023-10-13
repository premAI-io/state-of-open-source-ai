document.addEventListener('DOMContentLoaded', function(){
  /// download whole book PDF
  document.querySelectorAll('.btn-download-pdf-button').forEach(a => {
      a.onclick = () => window.open("/state-of-open-source-ai.pdf");
  });
  /// hide Markdown button
  document.querySelectorAll('.btn-download-source-button').forEach(a => {
    a.parentElement.style.display = 'none';
  });
});

document.addEventListener('DOMContentLoaded', function(){
  /// download whole book PDF
  document.querySelectorAll('.btn-download-pdf-button').forEach(a => {
      a.onclick = () => window.open("/state-of-open-source-ai.pdf");
  });
});

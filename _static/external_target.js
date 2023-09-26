document.addEventListener('DOMContentLoaded', function(){
  /// open external links in new tabs
  document.querySelectorAll('a.reference.external').forEach(a => {
    a.target = '_blank';
    if (a.href.startsWith("https://github.com/premAI-io/state-of-open-source-ai")){
      a.classList.replace('external', 'internal');
    }
  });
});

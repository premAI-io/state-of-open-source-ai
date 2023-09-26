document.addEventListener('DOMContentLoaded', function(){
  /// open external links in new tabs
  document.querySelectorAll('a.reference.external').forEach(a => { a.target = '_blank'; });
});

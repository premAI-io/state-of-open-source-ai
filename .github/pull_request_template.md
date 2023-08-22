# Review checklist

> Don't worry about satisfying all items, it's fine to open a (draft) PR.

- [ ] chapter content
  + [ ] only one top-level `# h1-Title`
  + [ ] summary (e.g. table or TL;DR overview), no need for an explicit `## Summary/Introduction` title or equivalent
  + [ ] main content focus: recent developments in open source AI
    + general context/background (brief)
    + current pros/cons
    + in-depth insights (not yet widely known)
  + [ ] likely `## Future` developments
  + [ ] end with `{{ comments }}`
- [ ] appropriate citations
  + [ ] BibTeX references
  + [ ] Glossary terms
  + [ ] cross-references (figures/chapters)
  + [ ] (if `new-chapter.md`), add `_toc.yml` entry & `index.md` table row
  + [ ] If CI URL checks have false-positives, append to `_config.yml:sphinx.config.linkcheck*`
- [ ] images & data not committed to this repo (e.g. use https://github.com/premAI-io/static.premai.io instead)

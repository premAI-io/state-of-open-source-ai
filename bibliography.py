"""Limit the number of authors shown in the bibliography."""
from pybtex.plugin import register_plugin
from sphinx.application import Sphinx
from pybtex.style.formatting import toplevel
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.formatting.unsrt import date, pages
from pybtex.style.template import (
    field, first_of, join, optional, optional_field, sentence, tag, words
)

__version__ = '0.0.0'


class Style(UnsrtStyle):

    def get_article_template(self, e):
        volume_and_pages = first_of [
            # volume and pages, with optional issue number
            optional [
                join [
                    field('volume'),
                    optional['(', field('number'),')'],
                    ':', pages
                ],
            ],
            # pages only
            words ['pages', pages],
        ]
        template = toplevel [
            self.format_names('author'),
            self.format_title(e, 'title'),
            sentence [
                tag('em') [field('journal')],
                optional[ volume_and_pages ],
                date],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def format_author(self, e, as_sentence = True, max_authors=5):
        authors = self.format_names('author', as_sentence=False)
        if 'author' not in e.persons:
            return authors
        if len(e.persons['author']) > max_authors:
            word = 'et al'
            e.persons['author'] = e.persons['author'][:max_authors]
            result = join(sep=', ')[authors, word]
        else:
            result = authors
        if as_sentence:
            return sentence[result]
        else:
            return result

    def format_author_or_editor(self, e):
        return first_of [
            optional[ self.format_author(e) ],
            self.format_editor(e),
        ]

    def get_booklet_template(self, e):
        template = toplevel [
            self.format_author(e),
            self.format_title(e, 'title'),
            sentence [
                optional_field('howpublished'),
                optional_field('address'),
                date,
                optional_field('note'),
            ],
            self.format_web_refs(e),
        ]
        return template

    def get_incollection_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_title(e, 'title'),
            words [
                'In',
                sentence [
                    optional[ self.format_editor(e, as_sentence=False) ],
                    self.format_btitle(e, 'booktitle', as_sentence=False),
                    self.format_volume_and_series(e, as_sentence=False),
                    self.format_chapter_and_pages(e),
                ],
            ],
            sentence [
                optional_field('publisher'),
                optional_field('address'),
                self.format_edition(e),
                date,
            ],
            self.format_web_refs(e),
        ]
        return template

    def get_inproceedings_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_title(e, 'title'),
            words [
                'In',
                sentence [
                    optional[ self.format_editor(e, as_sentence=False) ],
                    self.format_btitle(e, 'booktitle', as_sentence=False),
                    self.format_volume_and_series(e, as_sentence=False),
                    optional[ pages ],
                ],
                self.format_address_organization_publisher_date(e),
            ],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def get_mastersthesis_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_title(e, 'title'),
            sentence[
                "Master's thesis",
                field('school'),
                optional_field('address'),
                date,
            ],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def get_misc_template(self, e):
        template = toplevel [
            optional[ sentence [self.format_author(e)] ],
            optional[ self.format_title(e, 'title') ],
            sentence[
                optional[ field('howpublished') ],
                optional[ date ],
            ],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def get_online_template(self, e):
        return self.get_misc_template(e)

    def get_phdthesis_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_btitle(e, 'title'),
            sentence[
                first_of [
                    optional_field('type'),
                    'PhD thesis',
                ],
                field('school'),
                optional_field('address'),
                date,
            ],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def get_techreport_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_title(e, 'title'),
            sentence [
                words[
                    first_of [
                        optional_field('type'),
                        'Technical Report',
                    ],
                    optional_field('number'),
                ],
                field('institution'),
                optional_field('address'),
                date,
            ],
            sentence [ optional_field('note') ],
            self.format_web_refs(e),
        ]
        return template

    def get_unpublished_template(self, e):
        template = toplevel [
            sentence [self.format_author(e)],
            self.format_title(e, 'title'),
            sentence [
                field('note'),
                optional[ date ]
            ],
            self.format_web_refs(e),
        ]
        return template


def setup(app: Sphinx):
    register_plugin('pybtex.style.formatting', 'unsrt_max_authors', Style)
    return {'version': __version__}

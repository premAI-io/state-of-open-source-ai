"""Limit the number of authors shown in the bibliography."""
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import FieldIsMissing, join, node, sentence, tag
from sphinx.application import Sphinx

__version__ = '0.0.0'


@node
def names_truncated(children, context, role, max_names=9, **kwargs):
    """Return formatted names."""
    assert not children
    try:
        persons = context['entry'].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context['entry'])

    style = context['style']
    if (truncate := len(persons) > max_names):
        persons = persons[:max_names - 1]
    formatted_names = [style.format_name(person, style.abbreviate_names) for person in persons]
    if truncate:
        formatted_names.append(tag('i')["others"])
    return join(**kwargs)[formatted_names].format_data(context)


class Style(UnsrtStyle):
    def format_names(self, role, as_sentence=True):
        formatted_names = names_truncated(role, sep=', ', sep2=' and ', last_sep=', and ')
        return sentence[formatted_names] if as_sentence else formatted_names


def setup(app: Sphinx):
    register_plugin('pybtex.style.formatting', 'unsrt_max_authors', Style)
    return {'version': __version__}

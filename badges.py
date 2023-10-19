"""Display repository badges.

MyST usage (HTML only):

    ```{badges} https://mybook.site https://github.com/org/mybook
    :doi: 10.5281.zenodo.12345678
    ```
"""
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx

__version__ = '0.0.0'


def visit_nop(self, node):
    pass


class badges_node(nodes.General, nodes.Element):
    pass


def visit_badges_html(self, node):
    self.body.append(
        f"""<a href="{node['baseurl']}" target="_blank">
        <img alt="site"
         src="https://img.shields.io/badge/site-{node['baseurl'].replace('-', '--')}-orange" />
        </a>""")
    slug = '/'.join(node['repository_url'].split('/')[-2:])
    self.body.append(
        f"""<a href="{node['repository_url']}/graphs/contributors" target="_blank">
        <img alt="last updated"
         src="https://img.shields.io/github/last-commit/{slug}/main?label=updated" />
        </a>""")
    self.body.append(
        f"""<a href="{node['repository_url']}/pulse" target="_blank">
        <img alt="activity"
         src="https://img.shields.io/github/commit-activity/m/{slug}/main?label=commits" />
        </a>""")
    if node['doi']:
        self.body.append(
            f"""<a href="https://doi.org/{node['doi']}" target="_blank">
            <img alt="doi"
             src="https://img.shields.io/badge/doi-{node['doi']}-black" />
            </a>""")


class Badges(Directive):
    has_content = True
    required_arguments = 2
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {'doi': directives.unchanged}
    _node = None

    def run(self):
        return [badges_node(
            baseurl=self.arguments[0], repository_url=self.arguments[1], doi=self.options.get('doi', None))]


def setup(app: Sphinx):
    app.add_node(badges_node, html=(visit_badges_html, visit_nop),
                 latex=(visit_nop, visit_nop))
    app.add_directive("badges", Badges)
    return {'version': __version__, 'parallel_read_safe': True}

"""Display repository badges.

MyST usage (HTML only):

    ```{badges}
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
        f"""<a href="{node['baseurl']}">
        <img alt="site"
         src="https://img.shields.io/badge/site-{node['baseurl'].replace('-', '--')}-orange" />
        </a>""")
    slug = '/'.join(node['repository_url'].split('/')[-2:])
    self.body.append(
        f"""<a href="{node['repository_url']}/graphs/contributors">
        <img alt="last updated"
         src="https://img.shields.io/github/last-commit/{slug}/main" />
        </a>""")
    self.body.append(
        f"""<a href="{node['repository_url']}/pulse">
        <img alt="activity"
         src="https://img.shields.io/github/commit-activity/m/{slug}/main" />
        </a>""")


class Badges(Directive):
    has_content = True
    required_arguments = 2
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'class': directives.class_option, 'name': directives.unchanged}
    _node = None

    def run(self):
        return [badges_node(baseurl=self.arguments[0], repository_url=self.arguments[1])]


def setup(app: Sphinx):
    app.add_node(badges_node, html=(visit_badges_html, visit_nop),
                 latex=(visit_nop, visit_nop))
    app.add_directive("badges", Badges)
    return {'version': __version__, 'parallel_read_safe': True}

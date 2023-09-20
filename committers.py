"""Display Git committers & last updated time.

Example MyST usage (HTML only):

    ```{committers} file_path.md
    ```
"""
import json
import os
import re
import subprocess
from collections import Counter
from functools import cache
from urllib.request import Request, urlopen

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx

__version__ = '0.0.0'


@cache
def gh_api(endpoint: str, version='2022-11-28') -> dict:
    headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': version}
    if (token := os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))):
        headers['Authorization'] = 'Bearer ' + token  # higher rate limit & more permissions
    response = urlopen(Request("https://api.github.com/" + endpoint, headers=headers))
    return json.load(response)


def gh_user(email: str) -> str | None:
    if (user := {'het@hets-mbp.lan': 'htrivedi99', 'skanda.vivek@gmail.com': 'skandavivek'}.get(email, '')):
        return user  # hardcoded exceptions

    user_info = gh_api(f"search/users?q={email}+in:email")
    try:
        return user_info['items'][0]['login']
    except (KeyError, IndexError):
        return


class committers_node(nodes.General, nodes.Element):
    pass


def visit_nop(self, node):
    pass


def visit_committers_html(self, node):
    self.body.append(self.starttag(node, 'div'))
    self.body.append(f"Chapter author{'' if len(node['authors']) == 1 else 's'}: ")
    self.body.append(", ".join(f'<a href="{href}">{name}</a>' for name, href in node['authors']))
    self.body.append('</div>')


class Committers(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'class': directives.class_option, 'name': directives.unchanged}
    _node = None

    def run(self):
        blame = subprocess.check_output([
            'git', 'blame', '--line-porcelain', '-w', '-M', '-C', '-C', '--'] + self.arguments
        ).decode('utf-8').strip()
        authors = Counter(re.findall("^author (.*)\nauthor-mail <(.*)>", blame, flags=re.MULTILINE))
        total_loc = authors.total()
        auths = []
        for (name, email), loc in authors.most_common():
            if loc / total_loc < 0.1:  # ignore contributions under 10%
                break
            if (user := gh_user(email)):
                auths.append((name, f"https://github.com/{user}"))
            else:
                auths.append((name, f"mailto:{email}"))
        return [committers_node(authors=auths)]


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
    app.add_node(committers_node, html=(visit_committers_html, visit_nop),
                 latex=(visit_nop, visit_nop))
    app.add_directive("committers", Committers)
    app.add_node(badges_node, html=(visit_badges_html, visit_nop),
                 latex=(visit_nop, visit_nop))
    app.add_directive("badges", Badges)
    return {'version': __version__, 'parallel_read_safe': True}

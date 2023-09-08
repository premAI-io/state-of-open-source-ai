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
from datetime import timedelta
from functools import cache
from time import ctime, time

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx

__version__ = '0.0.0'


@cache
def gh_user(email) -> str | None:
    headers = [
        '--header', 'Accept: application/vnd.github+json',
        '--header', 'X-GitHub-Api-Version: 2022-11-28']
    if (token := os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))):
        headers.extend(['--header', f'Authorization: Bearer {token}'])
    for cmd in (
        ['gh', 'api'] + headers + [f'search/users?q={email}+in:email'],
        ['curl'] + headers + ['-fsSL', f'https://api.github.com/search/users?q={email}+in:email'],
        ['wget'] + headers + ['-qO', '-', f'https://api.github.com/search/users?q={email}+in:email']
    ):
        try:
            user_info = subprocess.check_output(cmd).decode('utf-8').strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        else:
            try:
                return json.loads(user_info)['items'][0]['login']
            except (KeyError, IndexError):
                return


class committers_node(nodes.General, nodes.Element):
    pass


def visit_nop(self, node):
    pass


def visit_committers_html(self, node):
    self.body.append(self.starttag(node, 'div'))
    self.body.append(f"Author{'' if len(node['authors']) == 1 else 's'}: ")
    self.body.append(", ".join(f'<a href="{href}">{name}</a>' for name, href in node['authors']))
    self.body.append("<br/>")
    self.body.append("Last updated: ")
    days = timedelta(seconds=time() - node['last_updated']).days
    self.body.append(f"{ctime(node['last_updated'])} ({days} day{'' if days == 1 else 's'} ago)")


def depart_committers_html(self, node):
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
            'git', 'blame', '--line-porcelain', '-C2', '-M2', '-w', '--'] + self.arguments
        ).decode('utf-8').strip()
        authors = Counter(re.findall("^author (.*)\nauthor-mail <(.*)>", blame, flags=re.MULTILINE))
        total_loc = authors.total()
        auths = []
        for (name, email), loc in authors.most_common():
            if loc / total_loc < 0.05:  # ignore contributions under 5%
                break
            if (user := gh_user(email)):
                auths.append((name, f"https://github.com/{user}"))
            else:
                auths.append((name, f"mailto:{email}"))
        updated = max(map(int, re.findall(r"^author-time (\d+)$", blame, flags=re.MULTILINE)))
        return [committers_node(authors=auths, last_updated=updated)]


def setup(app: Sphinx):
    app.add_node(committers_node, html=(visit_committers_html, depart_committers_html),
                 latex=(visit_nop, visit_nop))
    app.add_directive("committers", Committers)
    return {'version': __version__, 'parallel_read_safe': True}

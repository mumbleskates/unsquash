import argparse
from getpass import getpass
import json
import re
import sqlite3
import sys
from tqdm import tqdm

from github import Github
from dulwich.client import SSHGitClient
from dulwich.repo import Repo


class PullRequestDatabase:
    def __init__(self, db_path, github_repo_name, github_token):
        self.github_repo = Github(github_token).get_repo(github_repo_name)
        self.db_path = db_path
        self.db = None

    def __enter__(self):
        self.db = sqlite3.connect(self.db_path)
        self.db.executescript("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS pull_requests(
                id INTEGER PRIMARY KEY,
                commits_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS commits(
                id TEXT PRIMARY KEY,
                json TEXT NOT NULL
            ) WITHOUT ROWID;
        """)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        self.db = None

    def pull_request_commits(self, pull_request_id):
        """
        Returns (list of pull request commit ids, bool of whether this pr
        was cached.
        """
        try:
            [[json_data]] = self.db.execute("""
                SELECT commits_json FROM pull_requests WHERE id = ?;
            """, (pull_request_id,))
            return json.loads(json_data), True
        except ValueError:
            pass
        return self._fetch(pull_request_id), False

    def commit(self, commit_id):
        try:
            [[json_data]] = self.db.execute("""
                SELECT json FROM commits WHERE id = ?;
            """, (commit_id,))
        except ValueError:
            raise KeyError("commit not in database", commit_id)
        return json.loads(json_data)

    def _fetch(self, pull_request_id):
        github_pull_request = self.github_repo.get_pull(pull_request_id)
        commits = []

        def gen_commits():
            for commit in github_pull_request.get_commits():
                commits.append(commit.sha)
                raw_data = json.dumps(commit.raw_data)
                yield commit.sha, raw_data

        with self.db as cursor:
            # save all the commit data in the database
            cursor.executemany("""
                INSERT OR IGNORE INTO commits(id, json) VALUES (?, ?);
            """, gen_commits())

            # save the PR itself including its list of commits in the database
            cursor.execute("""
                INSERT INTO pull_requests(id, commits_json) VALUES (?, ?);
            """, (pull_request_id, json.dumps(commits)))

        return [c.encode() for c in commits]


def detect_github_squash_commit(commit):
    """
    Returns the pull request number if this looks like a squash commit,
    otherwise returns None.
    """
    if commit.committer != b'GitHub <noreply@github.com>':
        return None  # squashes are committed by github
    if len(commit.parents) > 1:
        return None  # squashes aren't merge commits
    pr = re.search(br'^[^\n]*\(#(\d+)\)\n', commit.message, flags=re.MULTILINE)
    return pr and int(pr.group(1))


def detect_original_commit(commit):
    """
    Returns the commit id of the original commit if this is an unsquashed
    commit, otherwise returns None.
    """
    match = re.search(rb'\nunsquashbot_original_commit=([0-9a-f]+)\n$',
                      commit.message,
                      re.MULTILINE)
    return match and match.group(1)


def map_unsquashed_branch(repo, head):
    # mapping from original commit id to unsquashed branch commit id
    unsquashed_mapping = {}
    num_rewritten = 0
    for walk in tqdm(repo.get_walker(head),
                     desc="mapping unsquash branch", units="commit"):
        original_commit_id = detect_original_commit(walk.commit)
        if original_commit_id:
            unsquashed_mapping[original_commit_id] = walk.commit.id
            num_rewritten += 1
        else:
            # not rewritten, commit is unchanged
            unsquashed_mapping[walk.commit.id] = walk.commit.id
    return unsquashed_mapping


def main():
    parser = argparse.ArgumentParser(description="Download pull request info")
    parser.add_argument("--repo", type=str, required=True,
                        help="The file path to the local git repo")
    parser.add_argument("--github_repo", type=str, required=True,
                        help="The repo/name of the project on github")
    parser.add_argument("--pr_db", type=str, default="pull_requests.db",
                        help="The file path to the pull requests cache")
    parser.add_argument("--squashed_branch", type=str, required=True,
                        help="The name of the branch to be unsquashed")
    parser.add_argument("--unsquashed_branch", type=str, required=True,
                        help="The name of the unsquashed branch to build or "
                             "maintain")
    parser.add_argument("--unsquashed_committer", type=str,
                        default="UnsquashBot <unsquashbot@example.com>",
                        help="The committer line for the bot")
    parser.add_argument("--token_file", type=str, default=None,
                        help="File to read the github token from")
    args = parser.parse_args()
    bot_email = args.unsquashed_committer.encode()

    repo = Repo(args.repo)

    try:
        unsquashed_head = repo.refs[
            f"refs/heads/{args.unsquashed_branch}".encode()]
        unsquashed_mapping = map_unsquashed_branch(repo, unsquashed_head)
    except KeyError:
        print("Unsquashed branch does not yet exist")
        unsquashed_head = None
        unsquashed_mapping = {}

    try:
        squashed_head = repo.refs[
            f"refs/heads/{args.squashed_branch}".encode()]
    except KeyError:
        print(f"Squashed branch {repr(args.squashed_branch)} not found!")
        sys.exit(1)

    if args.token_file is None:
        token = getpass(prompt="github token: ")
    else:
        with open(args.token_file, 'r') as f:
            token = f.read().strip()

    with PullRequestDatabase(
            db_path=args.pr_db,
            github_repo_name=args.github_repo,
            github_token=token) as pr_db:
        commit_stack = []
        expected_squash_commits = 0
        for walk in tqdm(repo.get_walker(squashed_head),
                         desc="crawling squashed branch", unit="commit"):
            if walk.commit.id not in unsquashed_mapping:
                commit_stack.append(walk.commit.id)
                if detect_github_squash_commit(walk.commit):
                    expected_squash_commits += 1
        with tqdm(total=len(commit_stack),
                  desc="unsquashing ", unit="commit") as rewrite_progress, \
                tqdm(total=expected_squash_commits,
                     desc="squashed prs", unit="pr") as pr_progress, \
                tqdm(desc="fetching commits",
                     unit="commit") as fetch_commit_progress:
            while commit_stack:
                current_commit_id = commit_stack.pop()
                current_commit = repo[current_commit_id]
                pull_request_id = detect_github_squash_commit(current_commit)
                if pull_request_id is not None:
                    # TODO: we are only fetching from the api for now
                    pr_commits, was_cached = (
                        pr_db.pull_request_commits(pull_request_id))
                    pr_progress.update(1)
                    if not was_cached:
                        fetch_commit_progress.update(len(pr_commits))
                        pr_progress.refresh()
                        fetch_commit_progress.refresh()
                    # TODO: rewrite_progress.total += len(pr_commits)
                # TODO: remap commits
                # elif all(unsquashed_mapping.get(parent, default=parent) == parent
                #          for parent in walk.commit.parents):
                #     continue  # this commit's exactly the same in unsquashed history

                # # build unsquashed commit object
                # rewrite = walk.commit.copy()
                # rewrite.message = b''.join([
                #     walk.commit.message,
                #     b'\n' * (not walk.commit.message.endswith(b'\n')),
                #     b'unsquashbot_original_commit=',
                #     walk.commit.id,
                #     b'\n',
                # ])
                # rewrite.committer = bot_email
                #
                # rewrite.parents = [
                #     unsquashed_mapping.get(parent, default=parent)
                #     for parent in walk.commit.parents
                # ]
                rewrite_progress.update()


if __name__ == '__main__':
    main()

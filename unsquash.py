from __future__ import annotations

import argparse
from base64 import b64decode
from datetime import datetime, timedelta
from getpass import getpass
import json
import re
import sqlite3
import sys
import time

from dulwich.objects import Blob, Commit, Tree
from dulwich.repo import Repo
from github import Github, RateLimitExceededException
from tqdm import tqdm

__doc__ = """
-- GitHub Unsquasher --
This tool rewrites git history for github-hosted repositories, rewriting the
history as if your pull request squash commits were all merges instead and
pulling in all the commits of those pull requests into this new branch.

While the commits themselves cannot be exactly reproduced, they were going to be
rewritten anyway, and the content of the repository is precisely the same. Every
commit, even if its branch was long deleted or never present in your copy of the
repo, has its files and trees exactly reproduced (assuming github's API is
willing to serve the complete data; if you have file versions over 100MB that
are not already in your repo, this might not work).

This tool can be rerun repeatedly, maintaining an unsquashed branch by updating
only new commits from the repository! This gives a complete accounting of what
actually happened, complete with a more comprehensive 'git blame' output for
all your files. Not only that, but checking out the unsquashed branch is almost
instant, since when it is up to date it refers to the exact same tree as the
unsquashed branch.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Unsquash squashed pull requests from a github-based repo.")
    parser.add_argument("--repo", required=True,
                        help="The file path to the local git repo.")
    parser.add_argument("--github_repo", required=True,
                        help="The repo/name of the project on github.")
    parser.add_argument("--token_file", default=None,
                        help="The file to read the github token from. If this "
                             "is not given, you will be asked to enter/paste "
                             "the token like a password when the tool runs.")
    parser.add_argument("--no_github", action='store_true', default=False,
                        help="Disable usage of the API, relying only on the "
                             "cache. The script will crash if any data needs "
                             "to be fetched from the API.")
    parser.add_argument("--pr_db", default="pull_requests.db",
                        help="The file path to the pull requests cache.")
    parser.add_argument("--squashed_branch", default=None,
                        help="The name of the branch to be unsquashed.")
    parser.add_argument("--squashed_ref", default=None,
                        help="The name of the ref to be unsquashed. Like "
                             "--squashed_branch, but specifies long-form refs "
                             "like 'refs/heads/master' instead of 'master'. "
                             "At most one of --squashed_branch and "
                             "--squashed_ref may be given.")
    parser.add_argument("--unsquashed_branch", default=None,
                        help="The name of the unsquashed branch to build or "
                             "maintain. Defaults to 'unsquash-' + the name of "
                             "the squashed branch.")
    parser.add_argument("--unsquashed_ref", default=None,
                        help="The name of the unsquashed ref to build or "
                             "maintain. Like --unsquashed_branch, but "
                             "specifies long-form refs, like "
                             "'refs/heads/unsquash-master' instead of "
                             "'unsquash-master'. At most one of "
                             "--unsquashed_branch and --unsquashed_ref may be "
                             "given.")
    parser.add_argument("--also_map_branch", action='append', default=[],
                        help="Additional branches to map before beginning the "
                             "unsquash process. All mapped commits will be "
                             "kept as-is, and also stand in for the commits "
                             "referred to by the 'unsquashbot_original_commit' "
                             "trailer value in the commit message. Helpful "
                             "when one branch is already unsquashed and you "
                             "want its commits to be reused for a second, "
                             "related branch; simply unsquash into the new "
                             "branch and list any already-unsquashed branches "
                             "you want to include as an --also_map_branch "
                             "argument.")
    parser.add_argument("--also_map_ref", action='append', default=[],
                        help="Like --also_map_branch, but specifies long-form "
                             "refs, like 'refs/heads/master' instead of "
                             "'master'.")
    parser.add_argument("--bot_email", 
                        default="unsquashbot@example.com",
                        help="The email address in the bot's committer line.")
    args = parser.parse_args()
    unsquashed_committer = f"UnsquashBot <{args.bot_email}>".encode()

    repo = Repo(args.repo)

    if args.unsquashed_branch is not None and args.unsquashed_ref is not None:
        print("Please provide at most one of --unsquashed_branch and "
              "--unsquashed_ref.")
        sys.exit(1)

    if args.squashed_branch is not None and args.squashed_ref is not None:
        print("Please provide at most one of --squashed_branch and "
              "--squashed_ref.")
        sys.exit(1)
    elif args.squashed_branch is None and args.squashed_ref is None:
        args.squashed_branch = "master"  # default to master branch

    if args.squashed_branch is not None:
        squashed_ref = f"refs/heads/{args.squashed_branch}".encode()
    else:
        squashed_ref = args.squashed_ref.encode()

    if args.unsquashed_branch is None and args.unsquashed_ref is None:
        # unsquashed ref defaults to a rename of the squashed branch
        if args.squashed_branch is None:
            print("When no --squashed_branch is given, one of "
                  "--unsquashed_branch and --unsquashed_ref must be given.")
            sys.exit(1)
        unsquashed_ref = f"refs/heads/unsquash-{args.squashed_branch}".encode()
    elif args.unsquashed_branch is not None:
        unsquashed_ref = f"refs/heads/{args.unsquashed_branch}".encode()
    else:
        unsquashed_ref = args.unsquashed_ref.encode()

    also_map_refs = [*(f"refs/heads/{also}".encode()
                       for also in args.also_map_branch),
                     *(also.encode() for also in args.also_map_ref)]

    try:
        squashed_head = repo.refs[squashed_ref]
    except KeyError:
        # we need the squashed ref, otherwise there's nothing to recreate
        print(f"Squashed ref {squashed_ref} not found!")
        sys.exit(1)

    if args.no_github:
        token = None
    else:
        if args.token_file is None:
            token = getpass(prompt="github token: ")
        else:
            with open(args.token_file, 'r') as f:
                token = f.read().strip()

    with GithubCache(
            db_path=args.pr_db,
            github_repo_name=args.github_repo,
            github_token=token) as gh_db:
        rebuild_history(repo=repo, gh_db=gh_db,
                        unsquashed_committer=unsquashed_committer,
                        squashed_head=squashed_head,
                        unsquashed_ref=unsquashed_ref,
                        also_map_refs=also_map_refs)


class GithubCache:
    def __init__(self, db_path: str, github_repo_name: str,
                 github_token: str | None):
        """
        Connect a cache to sqlite at db_path and connect to the github API.
        If token is None, the API will not be available.
        """
        if github_token is not None:
            self.github = Github(github_token)
            self.github_repo_name = github_repo_name
            while True:
                try:
                    self.github_repo = self.github.get_repo(github_repo_name)
                    break
                except RateLimitExceededException:
                    self._wait_for_rate_limit()

        self.db_path = db_path
        self.db = None

    def __enter__(self):
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("""PRAGMA journal_mode = WAL;""")
        with self.db as cursor:
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS pull_requests(
                    project TEXT NOT NULL,
                    id INTEGER NOT NULL,
                    pr_json TEXT NOT NULL,
                    commits_json TEXT NOT NULL,
                    PRIMARY KEY (project, id)
                );
                CREATE TABLE IF NOT EXISTS objects(
                    id TEXT PRIMARY KEY,
                    json TEXT NOT NULL
                ) WITHOUT ROWID;
            """)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        self.db = None

    def pull_request(self, pull_request_id: int) -> ((dict, list[bytes]), bool):
        """
        Returns ((pull request json, list of commit ids), bool of whether this
        pr was cached).
        """
        with self.db as cursor:
            try:
                [[pr_json, commits_json]] = cursor.execute("""
                    SELECT pr_json, commits_json FROM pull_requests
                    WHERE project = ? AND id = ?;
                """, (self.github_repo_name, pull_request_id))
                return (
                    (json.loads(pr_json),
                     [c_id.encode() for c_id in json.loads(commits_json)]),
                    True
                )
            except ValueError:
                pass
            return self._fetch_pr(pull_request_id, cursor), False

    def object(self, object_id: bytes) -> dict:
        try:
            [[json_data]] = self.db.execute("""
                SELECT json FROM objects WHERE id = ?;
            """, (object_id.decode(),))
            return json.loads(json_data)
        except ValueError:
            raise KeyError("commit not in database", object_id)

    def commit(self, commit_id: bytes) -> (dict, bool):
        """
        Returns (commit json, bool of whether this object was cached.
        """
        with self.db as cursor:
            try:
                return self.object(commit_id), True
            except KeyError:
                pass
            return self._fetch_commit(commit_id, cursor), False

    def tree(self, tree_id: bytes) -> (dict, bool):
        """
        Returns (tree json, bool of whether this object was cached.
        """
        with self.db as cursor:
            try:
                return self.object(tree_id), True
            except KeyError:
                pass
            return self._fetch_tree(tree_id, cursor), False

    def blob(self, blob_id: bytes) -> (dict, bool):
        """
        Returns (blob json, bool of whether this object was cached.
        """
        with self.db as cursor:
            try:
                return self.object(blob_id), True
            except KeyError:
                pass
            return self._fetch_blob(blob_id, cursor), False

    def _fetch_pr(self, pull_request_id: int, cursor) -> (dict, list[bytes]):
        """
        Fetches a PR and all it scommits and returns a tuple of (the pull
        request json, the list of the pr's commit ids).
        """
        while True:
            try:
                github_pull_request = self.github_repo.get_pull(pull_request_id)
                raw_data: dict = github_pull_request.raw_data
                break
            except RateLimitExceededException:
                self._wait_for_rate_limit()

        # the paginated list class in the github library is pretty fragile
        # against errors during iteration, so get the pages all at once
        while True:
            try:
                commits = [c.sha for c in github_pull_request.get_commits()]
                break
            except RateLimitExceededException:
                self._wait_for_rate_limit()

        # save the PR and its list of commit ids in the database
        cursor.execute("""
            INSERT INTO pull_requests(project, id, pr_json, commits_json)
            VALUES (?, ?, ?, ?);
        """, (
            self.github_repo_name,
            pull_request_id,
            json.dumps(raw_data),
            json.dumps(commits)
        ))

        return raw_data, [c.encode() for c in commits]

    def _fetch_commit(self, commit_id: bytes, cursor) -> dict:
        """
        Fetches a commit from the API and returns it.
        """
        while True:
            try:
                github_commit = self.github_repo.get_git_commit(
                    commit_id.decode())
                assert github_commit.sha.encode() == commit_id, (
                    f"bad data from github for commit {commit_id}")
                raw_data: dict = github_commit.raw_data
                break
            except RateLimitExceededException:
                self._wait_for_rate_limit()
        cursor.execute("""
            INSERT INTO objects(id, json) VALUES (?, ?);
        """, (github_commit.sha, json.dumps(raw_data)))
        return raw_data

    def _fetch_tree(self, tree_id, cursor) -> dict:
        """
        Fetches a tree from the API and returns it.
        """
        while True:
            try:
                github_tree = self.github_repo.get_git_tree(tree_id.decode())
                assert github_tree.sha.encode() == tree_id, (
                    f"bad data from github for tree {tree_id}")
                raw_data: dict = github_tree.raw_data
                break
            except RateLimitExceededException:
                self._wait_for_rate_limit()
        cursor.execute("""
            INSERT INTO objects(id, json) VALUES (?, ?);
        """, (github_tree.sha, json.dumps(raw_data)))
        return raw_data

    def _fetch_blob(self, blob_id, cursor) -> dict:
        """
        Fetches a blob from the API and returns it.
        """
        while True:
            try:
                github_blob = self.github_repo.get_git_blob(blob_id.decode())
                assert github_blob.sha.encode() == blob_id, (
                    f"bad data from github for blob {blob_id}")
                raw_data: dict = github_blob.raw_data
                break
            except RateLimitExceededException:
                self._wait_for_rate_limit()
        cursor.execute("""
            INSERT INTO objects(id, json) VALUES (?, ?);
        """, (github_blob.sha, json.dumps(raw_data)))
        return raw_data

    def _wait_for_rate_limit(self) -> None:
        rate_limit = self.github.get_rate_limit()
        wait_start = datetime.utcnow()
        time_to_wait = (rate_limit.core.reset - wait_start
                        + timedelta(seconds=10))
        with tqdm(desc="waiting for github API rate limit",
                  total=time_to_wait.total_seconds(), unit="s") as progress:
            while True:
                time.sleep(1)
                now = datetime.utcnow()
                waited = now - wait_start
                progress.n = min(waited.total_seconds(),
                                 time_to_wait.total_seconds())
                progress.refresh()
                if waited > time_to_wait:
                    break


def detect_github_squash_commit(commit: Commit) -> int | None:
    """
    Returns the pull request number if this looks like a squash commit,
    otherwise returns None.
    """
    if commit.committer != b"GitHub <noreply@github.com>":
        return None  # squashes are committed by github
    if len(commit.parents) > 1:
        return None  # squashes aren't merge commits
    pr = re.search(br'^[^\n]*\(#(\d+)\)\n', commit.message, flags=re.MULTILINE)
    return pr and int(pr.group(1))


def detect_original_commit(commit: Commit) -> bytes | None:
    """
    Returns the commit id of the original commit if this is an unsquashed
    commit, otherwise returns None.
    """
    match = re.search(rb"\nunsquashbot_original_commit=([0-9a-f]+)\n$",
                      commit.message, re.MULTILINE)
    return match and match.group(1)


def map_unsquashed(repo: Repo, heads: list[bytes]) -> dict[bytes, bytes]:
    # mapping from original commit id to unsquashed branch commit id
    unsquashed_mapping = {}
    num_rewritten = 0
    for walk in tqdm(repo.get_walker(include=heads),
                     desc="mapping unsquashed commits", unit="commit"):
        original_commit_id = detect_original_commit(walk.commit)
        if original_commit_id:
            unsquashed_mapping[original_commit_id] = walk.commit.id
            num_rewritten += 1
        else:
            # not rewritten, commit is unchanged
            unsquashed_mapping[walk.commit.id] = walk.commit.id
    return unsquashed_mapping


def recreate_commit(commit_json: dict) -> Commit:
    """
    Recreate a commit object approximately from github api json.
    """
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    commit = Commit()
    commit.parents = [p['sha'].encode() for p in commit_json['parents']]
    # leftover hack: adapt in case this is a "pull request commit" json object
    # instead of a "git commit" json object. they're equally good but the pull
    # request ones have a bunch of extra data in them
    if 'commit' in commit_json:
        commit_json = commit_json['commit']
    commit.message = commit_json['message'].encode()
    commit.author = (f"{commit_json['author']['name']} "
                     f"<{commit_json['author']['email']}>".encode())
    author_time = datetime.strptime(commit_json['author']['date'],
                                    date_format)
    commit.author_time = int(author_time.timestamp())
    commit.author_timezone = 0
    commit.committer = (f"{commit_json['committer']['name']} "
                        f"<{commit_json['committer']['email']}"
                        f">".encode())
    commit_time = datetime.strptime(commit_json['committer']['date'],
                                    date_format)
    commit.commit_time = int(commit_time.timestamp())
    commit.commit_timezone = 0
    commit.tree = commit_json['tree']['sha'].encode()
    return commit


def recreate_tree(tree_json: dict) -> Tree:
    """
    Recreate a tree object exactly from github api json.
    """
    assert not tree_json['truncated'], (f"Tree {tree_json['sha']} from github "
                                        f"was truncated!")
    tree = Tree()
    for tree_entry in tree_json['tree']:
        tree.add(name=tree_entry['path'].encode(),
                 mode=int(tree_entry['mode'], 8),
                 hexsha=tree_entry['sha'].encode())
    assert tree.id == tree_json['sha'].encode()
    return tree


def recreate_blob(blob_json: dict) -> Blob:
    """
    Recreate a blob object exactly from github api json.
    """
    blob = Blob()
    blob.data = b64decode(blob_json['content'])  # it's always base64 encoded
    assert blob.id == blob_json['sha'].encode()
    return blob


def download_tree(repo: Repo, gh_db: GithubCache, tree_id: bytes,
                  fetch_progress: tqdm) -> None:
    """
    Downlaod the tree and all its reachable objects into the repo from the
    Github JSON api.
    """
    gh_tree, was_cached = gh_db.tree(tree_id)
    if not was_cached:
        fetch_progress.update(1)
        fetch_progress.refresh()
    for entry in gh_tree['tree']:
        if entry['sha'].encode() in repo:
            continue
        if entry['type'] == 'blob':
            gh_blob, was_cached = gh_db.blob(entry['sha'].encode())
            if not was_cached:
                fetch_progress.update(1)
                fetch_progress.refresh()
            repo.object_store.add_object(recreate_blob(gh_blob))
        elif entry['type'] == 'tree':
            download_tree(repo, gh_db, entry['sha'].encode(), fetch_progress)
        elif entry['type'] == 'commit' and entry['mode'] == '160000':
            pass  # submodule, no action required
        else:
            assert False, (f"Unknown, absurd tree entry object type "
                           f"{entry['type']} with mode {entry['mode']} "
                           f"in tree {tree_id}")
    repo.object_store.add_object(recreate_tree(gh_tree))


def rebuild_history(repo: Repo, gh_db: GithubCache, unsquashed_committer: bytes,
                    squashed_head: bytes, unsquashed_ref: bytes,
                    also_map_refs: list[bytes]) -> None:
    map_heads = []
    for ref in [unsquashed_ref, *also_map_refs]:
        try:
            head = repo.refs[ref]
            map_heads.append(head)
            if squashed_head == (detect_original_commit(repo[head]) or head):
                print(f"Already up to date in {ref.decode()}")
                if ref != unsquashed_ref:
                    print("Updating unsquashed ref")
                    repo[unsquashed_ref] = head
                return
        except KeyError:
            print(f"Ref {ref.decode()} not found in the repo")

    if map_heads:
        unsquashed_mapping = map_unsquashed(repo=repo, heads=map_heads)
    else:
        unsquashed_mapping = {}

    commit_stack = []
    known_pull_requests = set()
    for walk in tqdm(repo.get_walker(squashed_head),
                     desc="crawling squashed branch", unit="commit"):
        if walk.commit.id not in unsquashed_mapping:
            commit_stack.append(walk.commit.id)
            pr_id = detect_github_squash_commit(walk.commit)
            if pr_id is not None:
                known_pull_requests.add(pr_id)

    head_commit_id = None
    rewrite_progress = tqdm(total=len(commit_stack),
                            desc="unsquashing ", unit="commit")
    pr_progress = tqdm(total=len(known_pull_requests),
                       desc="squashed prs", unit="pr")
    fetch_pr_progress = tqdm(desc="fetching PRs", unit="pr")
    fetch_obj_progress = tqdm(desc="fetching missing objects", unit="obj")

    def close_progress_bars():
        rewrite_progress.close()
        pr_progress.close()
        fetch_pr_progress.close()
        fetch_obj_progress.close()

    try:
        while commit_stack:
            current_commit_id = commit_stack.pop()
            must_rewrite = False
            try:
                current_commit = repo[current_commit_id]
                reconstructed = False
            except KeyError:
                # commit is not in the repo
                must_rewrite = True
                gh_json, was_cached = gh_db.commit(current_commit_id)
                if not was_cached:
                    fetch_obj_progress.update(1)
                    fetch_obj_progress.refresh()
                current_commit = recreate_commit(gh_json)
                reconstructed = True

            pull_request_id = detect_github_squash_commit(current_commit)
            if (
                    pull_request_id is not None
                    and pull_request_id not in known_pull_requests
            ):
                known_pull_requests.add(pull_request_id)
                pr_progress.total += 1

            # parents of this commit need to be processed first
            parents_to_enqueue = [
                parent_id for parent_id in current_commit.parents
                if parent_id not in unsquashed_mapping
            ]
            if parents_to_enqueue:
                commit_stack.append(current_commit_id)
                commit_stack.extend(parents_to_enqueue)
                # regress N commits
                rewrite_progress.total += len(parents_to_enqueue)
                continue

            if pull_request_id is not None:
                must_rewrite = True
                (pr_json, pr_commits), was_cached = (
                    gh_db.pull_request(pull_request_id))
                if not was_cached:
                    fetch_pr_progress.update(1)
                    fetch_pr_progress.refresh()
                # github also calls squash commits "merge commit"
                assert current_commit_id == pr_json['merge_commit_sha'].encode()
                # TODO: what if it isn't there? what if it doesn't match?

                # ensure that all the PR's commits exist beforehand
                merge_tip = pr_json['head']['sha'].encode()
                assert merge_tip == pr_commits[-1]
                if merge_tip not in unsquashed_mapping:
                    # these commits must be rewritten before we can write the
                    # unsquashed merge PR. we push the PR commit back on the
                    # stack and will revisit it again when its attendant
                    # commits are all in.
                    commit_stack.append(current_commit_id)
                    commit_stack.append(merge_tip)
                    rewrite_progress.total += 1  # regress 1 commit
                    continue
                assert all(pr_c in unsquashed_mapping for pr_c in pr_commits)

                # convert this PR into a merge commit
                current_commit.parents = [*current_commit.parents, merge_tip]
                current_commit.committer = unsquashed_committer
                pr_progress.update(1)

            # remap parent commits
            rewritten_parents = [
                unsquashed_mapping[p]
                for p in current_commit.parents
            ]
            if (
                    not must_rewrite
                    and current_commit.parents == rewritten_parents
            ):
                # this commit is exactly the same in unsquashed history
                unsquashed_mapping[current_commit_id] = current_commit_id
                head_commit_id = current_commit_id
                rewrite_progress.update(1)
                continue

            # this commit has changed, rewrite it!
            current_commit.parents = rewritten_parents
            if current_commit.tree not in repo:
                download_tree(repo, gh_db, current_commit.tree,
                              fetch_obj_progress)
            current_commit.message = b''.join([
                current_commit.message.rstrip(),
                b"\n\n",
                b"unsquashbot_reconstructed\n" * reconstructed,
                b"unsquashbot_original_commit=",
                current_commit_id,
                b"\n",
            ])
            # insert into the mapping with the commit's new id
            head_commit_id = current_commit.id
            unsquashed_mapping[current_commit_id] = head_commit_id
            # write the altered commit into the repo
            repo.object_store.add_object(current_commit)
            rewrite_progress.update(1)
    finally:
        close_progress_bars()
        if head_commit_id is not None:
            print("Updating unsquashed ref")
            repo.refs[unsquashed_ref] = head_commit_id


if __name__ == '__main__':
    main()

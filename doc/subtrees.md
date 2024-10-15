# Subtrees in this repository

- `feynman`: `https://github.com/meamy/feynman.git` `ara` [human](https://github.com/meamy/feynman)
- `mlvoqc`: `https://github.com/inQWIRE/mlvoqc.git` `main` [human](https://github.com/inQWIRE/mlvoqc)
- `quartz`: `https://github.com/quantum-compiler/quartz.git` `master` [human](https://github.com/quantum-compiler/quartz)
- `quartz/external/HiGHS`: `https://github.com/ERGO-Code/HiGHS.git` master [human](https://github.com/ERGO-Code/HiGHS)
- `queso`: `https://github.com/qqq-wisc/queso.git` `main` [human](https://github.com/qqq-wisc/queso)
- `quizx`: `https://github.com/zxcalc/quizx.git` `master` [human](https://github.com/zxcalc/quizx)
- `topt`: `https://github.com/Luke-Heyfron/TOpt.git` `master` [human](https://github.com/Luke-Heyfron/TOpt)


# General git subtree usage

- PATH is the subtree path on the FS, in the repo.
- SUBTREE_URL is the git remote URL.
- REMOTE is a shorthand remote name you pick.

Quick-and-dirty:
```
git subtree add --prefix $PATH $SUBTREE_URL main --squash
git subtree pull --prefix $PATH $SUBTREE_URL main --squash
```

Using remote names:
```
git remote add -f $REMOTE $SUBTREE_URL
git subtree add --prefix $PATH $REMOTE main --squash
git fetch tpope-vim-surround main
git subtree pull --prefix $PATH $REMOTE main --squash
```


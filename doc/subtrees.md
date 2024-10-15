# Subtrees in this repository

| Subtree prefix | Subtree (git remote) URL | Subtree branch | Project link |
|----------------|---------------|------------------|--------------|
| feynman | https://github.com/meamy/feynman.git | ara | [Feynman github](https://github.com/meamy/feynman) |
| mlvoqc | https://github.com/inQWIRE/mlvoqc.git | main | [mlvoqc github](https://github.com/inQWIRE/mlvoqc) |
| quartz | https://github.com/quantum-compiler/quarz.git | master | [Quartz github](https://github.com/quantum-compiler/quartz) |
| quartz/external/HiGHS | https://github.com/ERGO-Code/HiGHS.git | master | [HiGHS github](https://github.com/ERGO-Code/HiGHS) |
| queso | https://github.com/qqq-wisc/queso.git | main | [QUESO github](https://github.com/qqq-wisc/queso) |
| quizx | https://github.com/zxcalc/quizx.git | master | [QuiZX github](https://github.com/zxcalc/quizx) |
| topt | https://github.com/Luke-Heyfron/TOpt.git | master | [TOpt github](https://github.com/Luke-Heyfron/TOpt) |


# General git subtree usage

- PATH is the subtree path on the FS, in the repo.
- SUBTREE_URL is the git remote URL.
- SUBTREE_BRANCH is the git remote branch you want to pull.
- REMOTE is a shorthand remote name you pick.

Quick-and-dirty:

Add: `git subtree add --prefix $PATH $SUBTREE_URL $SUBTREE_BRANCH --squash`

Update: `git subtree pull --prefix $PATH $SUBTREE_URL $SUBTREE_BRANCH --squash`

Using remote names:

Add:
```
git remote add -f $REMOTE $SUBTREE_URL
git subtree add --prefix $PATH $REMOTE $SUBTREE_BRANCH --squash
```

Update:
```
git fetch tpope-vim-surround main
git subtree pull --prefix $PATH $REMOTE $SUBTREE_BRANCH --squash
```


# Overall Instructions

- Don't leave any code with TODO items - instead put it into a TODO.md where the code would be and include it in the markup, explaining what there is todo
- DO NOT exaggerate what is done and what is not, please be straightforward, if there was not time for something or it did not get done I need to know.
- Please summarize your work in the comments and always leave behind a checklist of things that you see as left to do in the comments.
- If I ask you to clean up, you may absolutely move files and folders to a bak/ folder in any directory. If it does not exist you can create one. Don't leave trash behind out of fear of not wanting to change my repo. I want my repo changed, that's why I'm coming to you for help.
- Code *should not* exceed a few hundred lines. If it does, please split it up into logical files with includes. This is especially true of files with complex logic. It is NOT an excuse to say the "code is complex and therefore long", if it is complex that's even more reason to split it into logical pieces.
- **DO NOT** duplicate filenames and directories, if you need a file or dir from somewhere else and it needs to be in both places, use a symlink.
- Always use the last few minutes of your time to CLEAN UP, that means mark any TODO items as listed above (in a file called TODO.md) and do not leave any unfinished code silently behind.
- please keep the repository logically organized, ALWAYS check where you are coding- does it make sense to be there? if not move it and work there.
- The end state of a repo should be reasonable:

```
include/
build/
src/
QUICKSTART.md
README.md
docs/    <---- ALL OTHER DOCS GO HERE
bak/
```

- You may add to the above but you must logically categorize it in a directory. For example having a src/python-lib and src/c-lib is reasonable, same for varoius topics or business logic like src/transfer  src/compute  for example are valid. But the repo should stay clean at absolutely all times. Otherwise the PR does not get merged in.



# Preferences

- Podman over docker
- python for anything that is not performance critical
- C or Rust for anything performance critical
- Typescript with React for any web UI stuff or if appropriate

# semvecpy 

Semvecpy is a repository for Semantic Vectors work in Python.

It is a research project. It includes some well-tested production-grade work, but it's up to
users to be aware of which parts this is.

## Useful Social Conventions

Feel free to clone, fork, and play with this repository. When it comes to merging code,
feel free to submit merge requests, and for frequent contributors, ask the Admins for
Maintain permissions. We tend to say yes.

Since it's a research project, we encourage work-in-progress and experimental code. 
This does sometimes lead to duplicated functionality, some variation in naming and code styles,
etc. 

To make things work together, code may be refactored. It's nice to discuss this with original 
authors, but for small changes, developers with suitable permissions are encouraged to just
go ahead. It follows that some changes may be unexpected to some authors. If 
it's a bother just sync your repository wherever you left off, feel free to fork, and we can 
discuss how to merge things back together later. Basically DON'T WORRY.

## For External Users

### Installation

Clone the repository, or install using `pip install git+https://github.com/semanticvectors/semvecpy`.
TODO: It would be nice to make this just `pip install semvecpy`. See https://packaging.python.org/tutorials/packaging-projects/.

### Imports

So far the package has been mainly designed for internal experiments rather than external reuse. If statements like `import vectors` or `import permutations` cause name clashes for you, please let us know.

###

## For Internal Developers

* Use Python 3+. We're making no efforts to be compatible with Python 2 or earlier. 
* Imports should work relative to the project ./semvecpy directory.
  * We recommend adding this ./semvecpy directory to your PYTHONPATH. 
  * If using anaconda, there's documentation saying you don't need to set PYTHONPATH
  (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/), but I'm not
  sure this works for package internal development. Please shout if you know another solution.
* Module names are preferred with underscore_separators, but there's no firm rule in place.
* Test for module `.../dir/foo.py` are in `.../dir/foo_test.y`.
  * This is one of the standard patterns, and it makes it particularly easy to see which modules
  already have dedicated tests, and whether these files should be moved / renamed if the modules
  they're testing are renamed.
  

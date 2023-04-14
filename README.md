# AML_Project1_IRLS

If LogisticRegression_with_IRLS return the `numpy.linalg.LinAlgError: Singular matrix` error, it probably means that X contains some column with only 0 (or an interaction between two kolumns is a column with only 0), which makes X not invertible. 

LogisticRegression_with_IRLS.py - source code
tests.ipynb - example function uses
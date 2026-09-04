"""Deep-learning challengers for Track K.

CPU-first and deliberately small: ten features and 40,125 training rows do not
support a large network, and a benchmark that needed a GPU could not run in CI.
Nothing here is imported by the production serving path.
"""

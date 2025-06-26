Dataset Post-processing
=======================

Utilities in :mod:`training.postprocessing` help analyze code entries and filter
records by cyclomatic complexity. They can be used after scraping to refine the
final dataset.

Example::

    from training.postprocessing import analyze_code_ast, filter_by_complexity

    stats = analyze_code_ast("def foo(x):\n    return x * 2")
    print(stats["complexities"])  # {'foo': 1}

    builder.enhance_with_clustering()
    builder.dataset = filter_by_complexity(builder.dataset, min_complexity=2)


Gitlab Snippets
===============

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('gitlab_snippets')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

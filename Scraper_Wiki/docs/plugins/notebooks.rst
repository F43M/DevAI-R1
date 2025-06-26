Notebooks
=========

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('notebooks')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

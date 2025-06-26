Code Extractor
==============

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('code_extractor')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

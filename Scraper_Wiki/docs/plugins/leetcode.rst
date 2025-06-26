Leetcode
========

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('leetcode')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

Npm Packages
============

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('npm_packages')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

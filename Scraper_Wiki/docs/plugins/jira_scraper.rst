Jira Scraper
============

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('jira_scraper')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

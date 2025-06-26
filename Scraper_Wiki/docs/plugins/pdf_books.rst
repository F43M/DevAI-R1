Pdf Books
=========

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('pdf_books')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

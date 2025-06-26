University Courses
==================

Example usage::

    from plugins import load_plugin, run_plugin

    plugin = load_plugin('university_courses')
    records = run_plugin(plugin, ['en'], ['Example'])
    print(records[:1])

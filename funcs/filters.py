from datetime import datetime
import humanize

def register_filters(app):
    @app.template_filter('datetimeformat')
    def datetimeformat(value, format='%Y-%m-%d %H:%M'):
        return datetime.fromtimestamp(value).strftime(format)

    @app.template_filter('filesizeformat')
    def filesizeformat(value):
        return humanize.naturalsize(value)
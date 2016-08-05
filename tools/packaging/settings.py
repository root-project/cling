# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import biplist
import os.path

application = defines.get('app', '/Applications/Cling.app')
size = defines.get('size', '70M')
appname = os.path.basename(application)

def icon_from_app(app_path):
    plist_path = os.path.join(app_path, 'Contents', 'Info.plist')
    plist = biplist.readPlist(plist_path)
    icon_name = plist['CFBundleIconFile']
    icon_root,icon_ext = os.path.splitext(icon_name)
    if not icon_ext:
        icon_ext = '.icns'
    icon_name = icon_root + icon_ext
    return os.path.join(app_path, 'Contents', 'Resources', icon_name)

format = defines.get('format', 'UDBZ')
size = defines.get('size', '100M')
files = [ application ]
symlinks = { 'Applications': '/Applications' }
icon = os.path.join(application, 'Contents', 'Resources', 'LLVM.icns')
icon_locations = {
    appname:        (140, 120),
    'Applications': (500, 120)
    }

background = 'builtin-arrow'

show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
sidebar_width = 180

# Window position in ((x, y), (w, h)) format
window_rect = ((100, 100), (640, 280))
default_view = 'icon-view'

# General view configuration
show_icon_preview = False

include_icon_view_settings = 'auto'
include_list_view_settings = 'auto'

# Icon view configuration

arrange_by = None
grid_offset = (0, 0)
grid_spacing = 100
scroll_position = (0, 0)
label_pos = 'bottom'
text_size = 16
icon_size = 128

# List view configuration

list_icon_size = 16
list_text_size = 12
list_scroll_position = (0, 0)
list_sort_by = 'name'
list_use_relative_dates = True
list_calculate_all_sizes = False,
list_columns = ('name', 'date-modified', 'size', 'kind', 'date-added')

list_column_widths = {
    'name': 300,
    'date-modified': 181,
    'date-created': 181,
    'date-added': 181,
    'date-last-opened': 181,
    'size': 97,
    'kind': 115,
    'label': 100,
    'version': 75,
    'comments': 300,
    }

list_column_sort_directions = {
    'name': 'ascending',
    'date-modified': 'descending',
    'date-created': 'descending',
    'date-added': 'descending',
    'date-last-opened': 'descending',
    'size': 'descending',
    'kind': 'ascending',
    'label': 'ascending',
    'version': 'ascending',
    'comments': 'ascending',
    }

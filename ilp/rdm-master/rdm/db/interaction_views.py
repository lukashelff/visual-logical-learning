'''
MySQL interaction views.

@author: Anze Vavpetic <anze.vavpetic@ijs.si>
'''
import json
from django.shortcuts import render
from .context import DBContext

def database_db_context(request, input_dict, output_dict, widget):
    con = input_dict['connection']
    find_con = input_dict['find_connections'] == 'true'
    initial_context = DBContext(con, find_connections=find_con)
    initial_target_cols = initial_context.cols[initial_context.target_table]
    cols_dump = json.dumps(initial_context.cols)
    return render(request, 'interactions/db_context.html', {'widget':widget,
                                                            'context': initial_context,
                                                            'connections' : dict(initial_context.connected),
                                                            'target_cols' : initial_target_cols,
                                                            'cols' : cols_dump})



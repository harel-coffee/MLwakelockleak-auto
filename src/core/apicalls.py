
import os
import zipfile
from androguard.misc import AnalyzeAPK, AnalyzeDex
from pandas import DataFrame

########################################################################
#                 Android API Calls related functions                  #
########################################################################


def list_calls_apks_in_dir(dir, l):
    """ Return a list with all API calls found in first l APK files in dir """
    calls = []
    for f in os.listdir(dir):
        if f.lower().endswith("apk") and l > 0:
            calls.append(list_calls(os.path.join(dir, f)))
            l -= 1
    return calls


def list_calls(file):
    """ Return a list with all API calls found in file (APK). Calls definition
        are reformatted as in java declarations.
    """
    apicalls = []
    a, d, dx = AnalyzeAPK(file)
    # for method in d.get_methods(): ##/////////////////////////////Orignal
    for method in d[0].get_methods(): ##////////////////////////////Changed
        for i in method.get_instructions():
            if i.get_name()[:6] == "invoke":
                # get method desc
                call = i.get_output(0).split(',')[-1].strip()
                # remove return value
                call = call[:call.index(')')+1]

                # split in class and method
                call = call.split('->')
                # print("invoke=", call)
                method_class = call[0]
                ins_method, params = call[1].split('(')
                # print(method_class, ins_method, params)
                params = ','.join(parse_parameters(params.replace(')', '')))
                # print("Params=",params)
                apicall = "{0}.{1}({2})".format(method_class,
                                                ins_method,
                                                params)
                # print("Api calls = ", apicall)
                apicalls.append(apicall)

    return apicalls


def list_calls_with_permissions(file, permission_map_file):
    """ List all API calls which require a permissions in file (according the
        mapping from Felt et al. CSS 2011 in APICalls.txt).
    """

    df = DataFrame.from_csv(permission_map_file, sep='\t')
    a, d, dx = AnalyzeAPK(file)
    for method in d.get_methods():
        for i in method.get_instructions():
            if i.get_name()[:6] == "invoke":
                # get method desc
                call = i.get_output(0).split(',')[-1].strip()
                # remove return value
                call = call[:call.index(')')+1]
                # split in class and method
                call = call.split('->')
                method_class = type(call[0])
                ins_method, params = call[1].split('(')
                params = ','.join(parse_parameters(params.replace(')', '')))
                apicall = "{0}.{1}({2})".format(method_class,
                                                ins_method,
                                                params)
                try:
                    print(df.ix[apicall]["Permission(s)"])
                    print(apicall)
                except:
                    pass


def parse_parameters(p):
    """ Parse and format parameters extracted from API
        calls found in smali code
    """
    types = ['S', 'B', 'D', 'F', 'I', 'J', 'Z', 'C']
    parameters = []
    buff = []
    i = 0
    while i < len(p):
        if p[i] == '[':
            buff.append(p[i])
        if p[i] in types:
            buff.append(p[i])
            parameters.append(''.join(buff))
            buff = []
        if p[i] == 'L':
            buff.append(p[i:][:p[i:].index(';')+1])
            parameters.append(''.join(buff))
            i += len(buff[0])-1
            buff = []
        i += 1
    # return [type(param) for param in parameters]##//////////////Orignal
    return [param for param in parameters]##/////////////////////Changed


def list_XREF(file):
    """ List all XREF in the methods of a binary """

    try:
        a, d, dx = AnalyzeAPK(file)
    except zipfile.BadZipfile:
        # if file is not an APK, may be a dex object
        d, dx = AnalyzeDex(file)

    for method in dx.get_methods():
        print(method.name)
        print("XREFfrom:",
              [m[1].name for m in method.get_xref_from()])
        print("XREFto:", [m[1].name for m in method.get_xref_to()])

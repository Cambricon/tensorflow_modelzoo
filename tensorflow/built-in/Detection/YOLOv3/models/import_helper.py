global pkg_dict
pkg_dict = {}

def add_pkg(key, val):
    pkg_dict[key] = val

def get_pkg(key, defaultVal=None):
    try:
        return pkg_dict[key]
    except KeyError:
        return defaultVal

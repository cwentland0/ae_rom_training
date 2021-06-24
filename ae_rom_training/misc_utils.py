import re


def get_vars_from_data(data_list, net_idx, var_net_idxs):
    data_input = []
    num_spatial_dims = data_list[0].ndim - 2
    for data_mat in data_list:
        if num_spatial_dims == 1:
            data_input.append(data_mat[:, var_net_idxs[net_idx], :])
        elif num_spatial_dims == 2:
            data_input.append(data_mat[:, var_net_idxs[net_idx], :, :])
        else:
            data_input.append(data_mat[:, var_net_idxs[net_idx], :, :, :])

    return data_input


def parse_value(expr):
    """
    Parse read text value into dict value
    """

    try:
        return eval(expr)
    except:
        return eval(re.sub("\s+", ",", expr))
    else:
        return expr


def parse_line(line):
    """
    Parse read text line into dict key and value
    """

    eq = line.find("=")
    if eq == -1:
        raise Exception()
    key = line[:eq].strip()
    value = line[eq + 1 : -1].strip()
    return key, parse_value(value)


def read_input_file(inputFile):
    """
    Read input file
    """

    # TODO: better exception handling besides just a pass

    read_dict = {}
    with open(inputFile) as f:
        contents = f.readlines()

    for line in contents:
        try:
            key, val = parse_line(line)
            read_dict[key] = val
            # convert lists to NumPy arrays
            # if (type(val) == list):
            # 	read_dict[key] = np.asarray(val)
        except:
            pass

    return read_dict


def catch_input(in_dict, in_key, default_val):

    default_type = type(default_val)
    try:
        # if NoneType passed as default, trust user
        if isinstance(default_type, type(None)):
            out_val = in_dict[in_key]
        else:
            out_val = default_type(in_dict[in_key])
    except:
        out_val = default_val

    return out_val

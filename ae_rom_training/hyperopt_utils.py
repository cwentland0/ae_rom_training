from hyperopt import hp
from hyperopt.pyll import scope


def read_hp_input(input_dict, param_name, defaults):

    # get value of param_name from input_dict
    # if it's a single value, just assign choice

    # get default value of param_name

    pass


def hp_expression(parameter_name, expression_type, input_list: list):
    """Generate HyperOpt expression.

    Does some basic error checking for various HyperOpt expressions.
    input_list has different interpretation depending on expression_type.

    Args:
        parameter_name: name of hyperparameter to generate a HyperOpt expression for.
        expression_type: HyperOpt expression type, e.g. "choice", "uniform", "loguniform", etc.
        input_list: list of parameters required to define a given expression, different for each expression_type.

    Returns:
        HyperOpt expression for parameter_name defined by expression_type and entries of input_list.
    """

    if expression_type == "choice":
        expression = hp.choice(parameter_name, input_list)
    elif expression_type == "uniform":
        assert len(input_list) == 2, "uniform expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.uniform(parameter_name, input_list[0], input_list[1])
    elif expression_type == "uniformint":
        assert len(input_list) == 2, "uniformint expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.uniformint(parameter_name, input_list[0], input_list[1])
    elif expression_type == "quniform":
        assert len(input_list) == 3, "quniform expression only accepts 3 inputs (" + parameter_name + ")"
        expression = hp.quniform(parameter_name, input_list[0], input_list[1], input_list[2])
    elif expression_type == "quniformint":
        assert len(input_list) == 3, "quniformint expression only accepts 3 inputs (" + parameter_name + ")"
        expression = scope.int(hp.quniform(parameter_name, input_list[0], input_list[1], input_list[2]))
    elif expression_type == "loguniform":
        assert len(input_list) == 2, "loguniform expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.loguniform(parameter_name, input_list[0], input_list[1])
    elif expression_type == "qloguniform":
        assert len(input_list) == 3, "qloguniform expression only accepts 3 inputs (" + parameter_name + ")"
        expression = hp.qloguniform(parameter_name, input_list[0], input_list[1], input_list[2])
    else:
        raise ValueError("Invalid or un-implemented HyperOpt expression_type: " + str(expression_type))

    return expression

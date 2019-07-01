#!/usr/bin/python
"""
Calculate the number of model parameters for RNN models.
"""
import argparse

def multiple_values(num_values,
                    greater_or_equal,
                    data_type=int):
    """
    source: https://github.com/awslabs/sockeye/blob/master/sockeye/arguments.py
    Returns a method to be used in argument parsing to parse a string of the form "<val>:<val>[:<val>...]" into
    a tuple of values of type data_type.
    :param num_values: Optional number of ints required.
    :param greater_or_equal: Optional constraint that all values should be greater or equal to this value.
    :param data_type: Type of values. Default: int.
    :return: Method for parsing.
    """

    def parse(value_to_check):
        if ':' in value_to_check:
            expected_num_separators = num_values - 1 if num_values else 0
            if expected_num_separators > 0 and (value_to_check.count(':') != expected_num_separators):
                raise argparse.ArgumentTypeError("Expected either a single value or %d values separated by %s" %
                                                 (num_values, ":"))
            values = tuple(map(data_type, value_to_check.split(":", num_values - 1)))
        else:
            values = tuple([data_type(value_to_check)] * num_values)
        if greater_or_equal is not None:
            if any((value < greater_or_equal for value in values)):
                raise argparse.ArgumentTypeError("Must provide value greater or equal to %d" % greater_or_equal)
        return values

    return parse

def int_greater_or_equal(threshold):
    """
    source: https://github.com/awslabs/sockeye/blob/master/sockeye/arguments.py
    Returns a method that can be used in argument parsing to check that the int argument is greater or equal to `threshold`.
    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """

    def check_greater_equal(value):
        value_to_check = int(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError("must be greater or equal to %d." % threshold)
        return value_to_check

    return check_greater_equal

def get_args():
    parser = argparse.ArgumentParser(description='The hyperparameter settings.')
    parser.add_argument('--bpe-symbols-src', type=int_greater_or_equal(1), required=True,
                        help='The number of bpe operations for source side.')
    parser.add_argument('--bpe-symbols-trg', type=int_greater_or_equal(1), required=True,
                        help='The number of bpe operations for target side.')
    parser.add_argument('--rnn-cell-type', choices=['lstm', 'gru'], required=True,
                        help='RNN cell type for encoder and decoder.')
    parser.add_argument('--num-layers', type=multiple_values(num_values=2, greater_or_equal=1), required=True,
                        help='Number of Number of layers for encoder & decoder. '
                             'Use "x:x" to specify separate values for encoder & decoder.')
    parser.add_argument('--num-embed', type=multiple_values(num_values=2, greater_or_equal=1), required=True,
                        help='Embedding size for source and target tokens. '
                             'Use "x:x" to specify separate values for src&tgt.')
    parser.add_argument('--rnn-num-hidden', type=int_greater_or_equal(1), required=True,
                        help='Number of RNN hidden units for encoder and decoder.')

    args = parser.parse_args()

    return args

def get_model_params(cell, ib, ob, sn, tn, se, te, h):
    param_dict = {} # {name of parameters: shape of param matrix}

    # enc2decinit parameters
    if cell == 'lstm':
        x = 2*tn
    else:
        x = tn
    for i in range(x):
        param_dict['decoder_rnn_enc2decinit_{0}_bias'.format(i)] = (h,)
        param_dict['decoder_rnn_enc2decinit_{0}_weight'.format(i)] = (h,h)

    # hidden
    param_dict['decoder_rnn_hidden_bias'] = (h,)
    param_dict['decoder_rnn_hidden_weight'] = (h, 2*h)

    # decoder_lx
    if cell == 'lstm':
        y = 4*h
    else:
        y = 3*h
    for i in range(tn):
        if i == 0:
            z = h+se
        else:
            z = h
        param_dict['decoder_rnn_l{0}_h2h_bias'.format(i)] = (y,)
        param_dict['decoder_rnn_l{0}_h2h_weight'.format(i)] = (y,h)
        param_dict['decoder_rnn_l{0}_i2h_bias'.format(i)] = (y,)
        param_dict['decoder_rnn_l{0}_i2h_weight'.format(i)] = (y,z)

    # birnn
    if cell == 'lstm':
        y = 2*h
    else:
        y = int(1.5 * h)
    param_dict['encoder_birnn_forward_l0_h2h_bias'] = (y,)
    param_dict['encoder_birnn_forward_l0_h2h_weight'] = (y, h/2)
    param_dict['encoder_birnn_forward_l0_i2h_bias'] = (y,)
    param_dict['encoder_birnn_forward_l0_i2h_weight'] = (y, se)
    param_dict['encoder_birnn_reverse_l0_h2h_bias'] = (y,)
    param_dict['encoder_birnn_reverse_l0_h2h_weight'] = (y, h/2)
    param_dict['encoder_birnn_reverse_l0_i2h_bias'] = (y,)
    param_dict['encoder_birnn_reverse_l0_i2h_weight'] = (y, se)

    # encoder_lx
    if sn > 1:
        if cell == 'lstm':
            y = 4*h
        else:
            y = 3*h
        for i in range(sn-1):
            param_dict['encoder_rnn_l{0}_h2h_bias'.format(i)] = (y,)
            param_dict['encoder_rnn_l{0}_h2h_weight'.format(i)] = (y, h)
            param_dict['encoder_rnn_l{0}_i2h_bias'.format(i)] = (y,)
            param_dict['encoder_rnn_l{0}_i2h_weight'.format(i)] = (y, h)

    # io
    param_dict['source_embed_weight'] = (ib, se)
    param_dict['target_embed_weight'] = (ob, se)
    param_dict['target_output_bias'] = (ob,)
    param_dict['target_output_weight'] = (ob, h)

    return param_dict

def get_num_params(cell, ib, ob, sn, tn, se, te, h):
    io_nparam = ib*se + ob*(1+te+h)
    if cell == 'lstm':
        nparam = h*(-4*h + 8*se + (8*sn+10*tn)*(1+h) + 1) + io_nparam
    else:
        nparam = h*(-int(2.5*h) + 6*se + (6*sn+7*tn)*(1+h) + 1) + io_nparam
    return nparam

def main():
    args = get_args()
    cell = args.rnn_cell_type
    sb = args.bpe_symbols_src
    tb = args.bpe_symbols_trg
    sn, tn = args.num_layers
    se, te = args.num_embed
    h = args.rnn_num_hidden

    ib = sb
    ob = tb

    param_dict = get_model_params(cell, ib, ob, sn, tn, se, te, h)
    nparam = get_num_params(cell, ib, ob, sn, tn, se, te, h)

    info = []
    for name, shape in sorted(param_dict.items()):
        info.append("{0}: {1}".format(name, shape))

    print("***RNN model***\n")
    print("Model parameters: {0}\n".format(",".join(info)))
    print("Total # of parameters: {0}\n".format(nparam))

if __name__ == '__main__':
    main()

        







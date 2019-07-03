#!/usr/bin/python
"""
Calculate the number of model parameters for RNN models.
"""
import argparse
import os
from collections import OrderedDict

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

def regular_file():
    """
    source: https://github.com/awslabs/sockeye/blob/master/sockeye/arguments.py
    Returns a method that can be used in argument parsing to check the argument is a regular file or a symbolic link,
    but not, e.g., a process substitution.
    :return: A method that can be used as a type in argparse.
    """

    def check_regular_file(value_to_check):
        value_to_check = str(value_to_check)
        if not os.path.isfile(value_to_check):
            raise argparse.ArgumentTypeError("must exist and be a regular file.")
        return value_to_check

    return check_regular_file


def get_args():
    parser = argparse.ArgumentParser(description='The hyperparameter settings.')
    parser.add_argument('--bpe-symbols-src', type=int_greater_or_equal(1), required=False,
                        help='The number of bpe operations for source side.')
    parser.add_argument('--bpe-symbols-trg', type=int_greater_or_equal(1), required=False,
                        help='The number of bpe operations for target side.')
    parser.add_argument('--train-bpe-src', type=regular_file(), required=False,
                        help='Source side of parallel training data.')
    parser.add_argument('--train-bpe-trg', type=regular_file(), required=False,
                        help='Target side of parallel training data.')
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
    parser.add_argument('--exact', action='store_true', 
                        help='If specified as True, return the exact number of parameters with training data given.'
                             'Otherwise, return an approximate value with bpe-symbols given.')

    args = parser.parse_args()

    if args.exact:
        if ('train_bpe_src' not in vars(args)):
            parser.error('--train-bpe-src is required for exact calculation.')
        if ('train_bpe_trg' not in vars(args)):
            parser.error('--train-bpe-trg is required for exact calculation.')
    else:
        if ('bpe_symbols_src' not in vars(args)):
            parser.error('--bpe-symbols-src is required for approximate calculation.')
        if ('bpe_symbols_trg' not in vars(args)):
            parser.error('--bpe-symbols-trg is required for approximate calculation.')

    return args

def get_num_vocab(fname):
    '''
    Get the size of vocabulary.
    :param fname: BPE'ed training data.
    '''
    tokens = []
    with open(fname) as f:
        for line in f:
            for t in line.rstrip().split():
                if len(t) > 0:
                    tokens.append(t)
    return len(set(tokens))+4

def get_model_params(cell, sb, tb, sn, tn, se, te, h):
    group_dict = OrderedDict() # {name of parameter groups: number of parameters in the group}
    param_dict = {} # {name of parameters: shape of param matrix}

    # enc2decinit parameters
    if cell == 'lstm':
        x = 2*tn
        group_dict["enc2decinit"] = 2*tn*h*(h+1)
    else:
        x = tn
        group_dict["enc2decinit"] = tn*h*(h+1)
    for i in range(x):
        param_dict['decoder_rnn_enc2decinit_{0}_bias'.format(i)] = (h,)
        param_dict['decoder_rnn_enc2decinit_{0}_weight'.format(i)] = (h,h)


    # hidden
    param_dict['decoder_rnn_hidden_bias'] = (h,)
    param_dict['decoder_rnn_hidden_weight'] = (h, 2*h)

    group_dict["hidden"] = h*(2*h+1)

    # decoder_lx
    if cell == 'lstm':
        y = 4*h
        group_dict["decoder_lx"] = 4*h*(se+2*tn*(h+1))
    else:
        y = 3*h
        group_dict["decoder_lx"] = 3*h*(se+2*tn*(h+1))
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
        group_dict["birnn"] = 2*h*(4+h+2*se)
    else:
        y = int(1.5 * h)
        group_dict["birnn"] = int(1.5*h*(4+h+2*se))
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
            group_dict["encoder_lx"] = 4*h*(sn-1)*(2+2*h)
        else:
            y = 3*h
            group_dict["encoder_lx"] = 3*h*(sn-1)*(2+2*h)
        for i in range(sn-1):
            param_dict['encoder_rnn_l{0}_h2h_bias'.format(i)] = (y,)
            param_dict['encoder_rnn_l{0}_h2h_weight'.format(i)] = (y, h)
            param_dict['encoder_rnn_l{0}_i2h_bias'.format(i)] = (y,)
            param_dict['encoder_rnn_l{0}_i2h_weight'.format(i)] = (y, h)

    # io
    param_dict['source_embed_weight'] = (sb, se)
    param_dict['target_embed_weight'] = (tb, se)
    param_dict['target_output_bias'] = (tb,)
    param_dict['target_output_weight'] = (tb, h)

    group_dict["io"] = sb*se+tb*(1+te+h)

    return group_dict, param_dict

def get_num_params(cell, sb, tb, sn, tn, se, te, h):
    io_nparam = sb*se + tb*(1+te+h)
    if cell == 'lstm':
        nparam = h*(-4*h + 8*se + (8*sn+10*tn)*(1+h) + 1) + io_nparam
    else:
        nparam = h*(-int(2.5*h) + 6*se + (6*sn+7*tn)*(1+h) + 1) + io_nparam
    return nparam

def main():
    args = get_args()
    cell = args.rnn_cell_type
    sn, tn = args.num_layers
    se, te = args.num_embed
    h = args.rnn_num_hidden

    if args.exact:
        sb = get_num_vocab(args.train_bpe_src)
        tb = get_num_vocab(args.train_bpe_trg)
    else:
        sb = args.bpe_symbols_src
        tb = args.bpe_symbols_trg

    group_dict, param_dict = get_model_params(cell, sb, tb, sn, tn, se, te, h)
    nparam = get_num_params(cell, sb, tb, sn, tn, se, te, h)

    info = []
    for name, shape in sorted(param_dict.items()):
        info.append("{0}: {1}".format(name, shape))

    print("***RNN model***\n")
    print("Model parameters: {0}\n\n".format(",".join(info)))
    for group, num in group_dict.items():
        print("# of {0} parameters: {1}".format(group, num))
    print("\nTotal # of parameters: {0}\n".format(nparam))

if __name__ == '__main__':
    main()

        







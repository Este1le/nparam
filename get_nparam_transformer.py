#!/usr/bin/python
"""
Calculate the number of model parameters for Transformer models.
"""
import argparse
import os

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
    parser.add_argument('--num-layers', type=multiple_values(num_values=2, greater_or_equal=1), required=True,
                        help='Number of Number of layers for encoder & decoder. '
                             'Use "x:x" to specify separate values for encoder & decoder.')
    parser.add_argument('--num-embed', type=multiple_values(num_values=2, greater_or_equal=1), required=True,
                        help='Embedding size for source and target tokens. '
                             'Use "x:x" to specify separate values for src&tgt.')
    parser.add_argument('--transformer-model-size', type=int_greater_or_equal(1), required=True,
                        help='Number of hidden units in transformer layers.')
    parser.add_argument('--transformer-feed-forward-num-hidden', type=int_greater_or_equal(1), required=True,
                        help='Number of hidden units in transformers feed forward layers.')
    parser.add_argument('--exact', action='store_true', 
                        help='If specified as True, return the exact number of parameters with training data given.'
                             'Otherwise, return an approximate value with bpe-symbols given.')

    args = parser.parse_args()

    if args.num_embed[0] != args.transformer_model_size or args.num_embed[1] != args.transformer_model_size:
        parser.error("Transformer model size should be equal to the number of embedding.")

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

def get_model_params(sb, tb, sn, tn, e, f):
    param_dict = {} # {name of parameters: shape of param matrix}

    # decoder_att
    for i in range(tn):
        param_dict['decoder_transformer_{0}_att_enc_h2o_weight'.format(i)] = (e, e)
        param_dict['decoder_transformer_{0}_att_enc_k2h_weight'.format(i)] = (e, e)
        param_dict['decoder_transformer_{0}_att_enc_pre_norm_beta'.format(i)] = (e,)
        param_dict['decoder_transformer_{0}_att_enc_pre_norm_gamma'.format(i)] = (e,)
        param_dict['decoder_transformer_{0}_att_enc_q2h_weight'.format(i)] = (e, e)
        param_dict['decoder_transformer_{0}_att_enc_v2h_weight'.format(i)] = (e, e)
        param_dict['decoder_transformer_{0}_att_self_h2o_weight'.format(i)] = (e, e)
        param_dict['decoder_transformer_{0}_att_self_i2h_weight'.format(i)] = (3*e, e)
        param_dict['decoder_transformer_{0}_att_self_pre_norm_beta'.format(i)] = (e,)
        param_dict['decoder_transformer_{0}_att_self_pre_norm_gamma'.format(i)] = (e, )

    # decoder_ff
    for i in range(tn):
        param_dict['decoder_transformer_{0}_ff_h2o_bias'.format(i)] = (e,) 
        param_dict['decoder_transformer_{0}_ff_h2o_weight'.format(i)] = (e, f) 
        param_dict['decoder_transformer_{0}_ff_i2h_bias'.format(i)] = (f,) 
        param_dict['decoder_transformer_{0}_ff_i2h_weight'.format(i)] = (f, e) 
        param_dict['decoder_transformer_{0}_ff_pre_norm_beta'.format(i)] = (e,) 
        param_dict['decoder_transformer_{0}_ff_pre_norm_gamma'.format(i)] = (e,)

    # decoder_final
    param_dict['decoder_transformer_final_process_norm_beta'] = (e,)
    param_dict['decoder_transformer_final_process_norm_gamma'] = (e,)

    # encoder_att
    for i in range(sn):
        param_dict['encoder_transformer_{0}_att_self_h2o_weight'.format(i)] = (e, e) 
        param_dict['encoder_transformer_{0}_att_self_i2h_weight'.format(i)] = (3*e, e) 
        param_dict['encoder_transformer_{0}_att_self_pre_norm_beta'.format(i)] = (e,) 
        param_dict['encoder_transformer_{0}_att_self_pre_norm_gamma'.format(i)] = (e,)

    # encoder_ff
    for i in range(sn):
        param_dict['encoder_transformer_{0}_ff_h2o_bias'.format(i)] = (e,) 
        param_dict['encoder_transformer_{0}_ff_h2o_weight'.format(i)] = (e, f) 
        param_dict['encoder_transformer_{0}_ff_i2h_bias'.format(i)] = (f,) 
        param_dict['encoder_transformer_{0}_ff_i2h_weight'.format(i)] = (f, e) 
        param_dict['encoder_transformer_{0}_ff_pre_norm_beta'.format(i)] = (e,) 
        param_dict['encoder_transformer_{0}_ff_pre_norm_gamma'.format(i)] = (e,)

    # encoder_final
    param_dict['encoder_transformer_final_process_norm_beta'] = (e,)
    param_dict['encoder_transformer_final_process_norm_gamma'] = (e,)

    # io
    param_dict['source_embed_weight'] = (sb, e)
    param_dict['target_embed_weight'] = (tb, e)
    param_dict['target_output_bias'] = (tb,)
    param_dict['target_output_weight'] = (tb, e)

    return param_dict

def get_num_params(sb, tb, sn, tn, e, f):
    io_nparam = sb*e + tb*(2*e+1)
    nparam = tn*(8*e*e+7*e+2*e*f+f) + sn*(4*e*e+5*e+2*e*f+f) + 4*e + io_nparam

    return nparam

def main():
    args = get_args()
    sn, tn = args.num_layers
    se, te = args.num_embed
    e = se
    f = args.transformer_feed_forward_num_hidden

    if args.exact:
        sb = get_num_vocab(args.train_bpe_src)
        tb = get_num_vocab(args.train_bpe_trg)
    else:
        sb = args.bpe_symbols_src
        tb = args.bpe_symbols_trg

    param_dict = get_model_params(sb, tb, sn, tn, e, f)
    nparam = get_num_params(sb, tb, sn, tn, e, f)

    info = []
    for name, shape in sorted(param_dict.items()):
        info.append("{0}: {1}".format(name, shape))

    print("***Transformer model***\n")
    print("Model parameters: {0}\n".format(",".join(info)))
    print("Total # of parameters: {0}\n".format(nparam))

if __name__ == '__main__':
    main()

        







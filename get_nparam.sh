#！/bin/bash
#
# Calculate the number of model parameters given the hyperparameter settings

function errcho() {
	>&2 echo $1
}

function show_help() {
	errcho "Usage: get_nparam_rnn.sh -p hyperparams.txt"
	errcho ""
}

function check_file_exists() {
	if [ ! -f $1 ]; then
		errcho "FATAL: Could not find file $1"
		exit 1
	fi
}

while getopts ":h?p:" opt; do
	case "$opt" in
    	h|\?)
      	show_help
      	exit 0
      	;;
    	p) HYP_FILE=$OPTARG
      	;;
    esac
done

if [[ -z $HYP_FILE ]]; then
	errcho "Missing arguments"
	show_help
	exit 1
fi

check_file_exists $HYP_FILE
source $HYP_FILE

if [[ "$encoder" == "rnn" ]]; then
	python ./get_nparam_rnn.py --bpe-symbols-src ${bpe_symbols_src} \
				   --bpe-symbols-trg ${bpe_symbols_trg} \
				   --train-bpe-src ${train_bpe_src} \
				   --train-bpe-trg ${train_bpe_trg} \
				   --rnn-cell-type ${rnn_cell_type} \
				   --num-layers ${num_layers} \
				   --num-embed ${num_embed} \
				   --rnn-num-hidden ${rnn_num_hidden} \
				   --exact

elif [[ "$encoder" == "transformer" ]]; then
	python ./get_nparam_transformer.py --bpe-symbols-src ${bpe_symbols_src} \
					   --bpe-symbols-trg ${bpe_symbols_trg} \
					   --train-bpe-src ${train_bpe_src} \
					   --train-bpe-trg ${train_bpe_trg} \
					   --num-layers ${num_layers} \
					   --num-embed ${num_embed} \
					   --transformer-model-size ${transformer_model_size} \
					   --transformer-feed-forward-num-hidden ${transformer_feed_forward_num_hidden} \
					   --exact
else
	errcho "Input model should be either an RNN or a Transformer."
fi

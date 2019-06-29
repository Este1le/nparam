# nparam

For neural machine translation models built through [Amazon Sockeye](https://github.com/awslabs/sockeye), this documentation and the scripts show a way to calculate the number of model parameters by hand given hyperparameter settings. 

We consider RNN and Transformer models seperately. 

## RNN
### 1. Hyper-parameters

* `bpe_symbols`: 
* `rnn_cell_type`: RNN cell type for encoder and decoder, including `gru` and `lstm`.
* `num_layers`: Number of layers for encoder and decoder.
* `num_embed`: Embedding size for source and target tokens. 
* `rnn_num_hidden`: Number of RNN hidden units for encoder and decoder.

### 2. Parameters

The parameters (`parameter name: shape`) for a RNN model trained with `bpe_symbols=50000`, `rnn_cell_type=lstm`, `num_embed=512:512`, `rnn_num_hidden=512`, `num_layers=2:2` are shown below.

**enc2decinit**

```
decoder_rnn_enc2decinit_0_bias: (512,), 
decoder_rnn_enc2decinit_0_weight: (512, 512), 
decoder_rnn_enc2decinit_1_bias: (512,), 
decoder_rnn_enc2decinit_1_weight: (512, 512),
decoder_rnn_enc2decinit_2_bias: (512,), 
decoder_rnn_enc2decinit_2_weight: (512, 512), 
decoder_rnn_enc2decinit_3_bias: (512,), 
decoder_rnn_enc2decinit_3_weight: (512, 512) 
```

**hidden**

```
decoder_rnn_hidden_bias: (512,), 
decoder_rnn_hidden_weight: (512, 1024)
```

**decoder\_lx**

``` 
decoder_rnn_l0_h2h_bias: (2048,), 
decoder_rnn_l0_h2h_weight: (2048, 512), 
decoder_rnn_l0_i2h_bias: (2048,), 
decoder_rnn_l0_i2h_weight: (2048, 1024), 
decoder_rnn_l1_h2h_bias: (2048,), 
decoder_rnn_l1_h2h_weight: (2048, 512), 
decoder_rnn_l1_i2h_bias: (2048,), 
decoder_rnn_l1_i2h_weight: (2048, 512) 
```
**birnn**

```
encoder_birnn_forward_l0_h2h_bias: (1024,), 
encoder_birnn_forward_l0_h2h_weight: (1024, 256), 
encoder_birnn_forward_l0_i2h_bias: (1024,), 
encoder_birnn_forward_l0_i2h_weight: (1024, 512), 
encoder_birnn_reverse_l0_h2h_bias: (1024,), 
encoder_birnn_reverse_l0_h2h_weight: (1024, 256), 
encoder_birnn_reverse_l0_i2h_bias: (1024,), 
encoder_birnn_reverse_l0_i2h_weight: (1024, 512) 
```
**encoder\_lx**

```
encoder_rnn_l0_h2h_bias: (2048,), 
encoder_rnn_l0_h2h_weight: (2048, 512), 
encoder_rnn_l0_i2h_bias: (2048,), 
encoder_rnn_l0_i2h_weight: (2048, 512)
```
**io**

```
source_embed_weight: (49410, 512), 
target_embed_weight: (42767, 512), 
target_output_bias: (42767,), 
target_output_weight: (42767, 512)
```

The total number of parameters is calculated as follows:

```
total_num_params = sum([reduce(lambda s1,s2: s1*s2, shape) for param_name, shape in params.items()])
```

For the model above, the total number of parameters is 79638799.

### 3. Influence of Hyper-parameters on Parameters

Now let's see how the changes on each hyper-parameter reflect on the shape and number of parameter matrices.

Suppose `bpe_symbols=b`, `num_layers=sn:tn`, `num_embed=se:te`, `rnn_num_hidden=h`, where `s` stands for encoder, `t` stands for decoder. And `s1`, `s2` are the length of the first and second dimension of the parameter matrix.

* `bpe_symbols`

* `rnn_cell_type`

	**enc2decinit**: Lstm has `decoder_rnn_enc2decinit_x_...`, where `x=0,...,2*n-1`; while for gru, `x=0,...,n-1`.
	
	**decoder\_lx**, **encoder\_lx**: For lstm, `s1=4*h`; while for gru, `s1=3*h`.
	
	**birnn**: For lstm, `s1=2*h`; while for gru, `s1=3*h/2`. 

* `num_layers`
	
	**enc2decinit**: Lstm has `decoder_rnn_enc2decinit_x_bias/weight`, where `x=0,...,2*n-1`; while for gru, `x=0,...,n-1`.
	
	**decoder\_lx**: `decoder_rnn_lx_...`, where `x=0,...,n-1`.
	
	**encoder\_lx**: `encoder_rnn_lx_...`, where `x=0,...,n-2`. Notice when `n=1`, these parameters do not exist.

* `num_embed`
	
	**birnn**: For `encoder_birnn_forward_lx_i2h_weight` and `encoder_birnn_reverse_lx_i2h_weight`, `s2=e`.
	
	**io**: For `source_embed_weight` and `target_embed_weight`, `s2=e`.

* `rnn_num_hidden`
	
	**enc2decinit**: `s1=h`. For `weight` parameters, `s2=h` .
	
	**hidden**: `s1=h`. For `weight` parameters, `s2=2*h` .
	
	**decoder\_lx**: `s1=4*h`. For `weight` parameters, `s2=h`, except for `decoder_rnn_l0_i2h_weight`, where `s2=?`.

	**birnn**: `s1=2*h`. For `h2h_weight` parameters, `s2=h/2`, for `i2h_weight` parameters, `s2=?`.
	
	**encoder\_lx**: `s1=4*h`. For `weight` parameters, `s2=h`.
	
	**io**: For `target_output_weight` parameters, `s2=h`.
	
	
	
<<<<<<< HEAD
### 4. Parameters wrt Hyper-parameters

From previous section, we get the equation for calculating the number of parameters based on hyper-parameter settings.

Suppose `bpe_symbols=b`, `num_layers=sn:tn`, `num_embed=se:te`, `rnn_num_hidden=h`.




	
=======
	**encoder\_lx**: `encoder_rnn_lx_...`, where `x=0,...,n-2`. Notice when `n=1`, these parameters do not exist.
	
>>>>>>> b01c89b640d244f6ab98c65f6dab2a83ac5e85b4

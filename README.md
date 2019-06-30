# nparam

For neural machine translation models built through [Amazon Sockeye](https://github.com/awslabs/sockeye), this documentation and the scripts show a way to calculate the number of model parameters by hand given hyperparameter settings. 

We will consider RNN and Transformer models seperately. 

## RNN
### 1. Hyper-parameters

* `bpe_symbols`: 
* `rnn_cell_type`: RNN cell type for encoder and decoder, including `gru` and `lstm`.
* `num_layers`: Number of layers for encoder and decoder.
* `num_embed`: Embedding size for source and target tokens. 
* `rnn_num_hidden`: Number of RNN hidden units for encoder and decoder.

### 2. Parameters

The parameters (`parameter name: shape`) for a RNN model trained with `bpe_symbols=50000`, `rnn_cell_type=lstm`, `num_embed=512:512`, `rnn_num_hidden=512`, `num_layers=2` are shown below.

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

Suppose `bpe_symbols=sb:tb`, `num_layers=sn:tn`, `num_embed=se:te`, `rnn_num_hidden=h`, where `s` stands for encoder, `t` stands for decoder. And `s1`, `s2` are the length of the first and second dimension of the parameter matrix.

* `bpe_symbols`

	?

* `rnn_cell_type`

	**enc2decinit**: Lstm has `decoder_rnn_enc2decinit_x_...`, where `x=0,...,2*tn-1`; while for gru, `x=0,...,tn-1`.
	
	**decoder\_lx**, **encoder\_lx**: For lstm, `s1=4h`; while for gru, `s1=3h`.
	
	**birnn**: For lstm, `s1=2h`; while for gru, `s1=3h/2`. 

* `num_layers`
	
	**enc2decinit**: Lstm has `decoder_rnn_enc2decinit_x_bias/weight`, where `x=0,...,2*tn-1`; while for gru, `x=0,...,tn-1`.
	
	**decoder\_lx**: `decoder_rnn_lx_...`, where `x=0,...,tn-1`.
	
	**encoder\_lx**: `encoder_rnn_lx_...`, where `x=0,...,sn-2`. Notice when `sn=1`, these parameters do not exist.

* `num_embed`
	
	**decoder\_lx**: For `decoder_rnn_l0_i2h_weight`, `s2=h+se`.
	
	**birnn**: For `i2h_weight`, `s2=se`.
	
	**io**: For `source_embed_weight`, `s2=se` and for `target_embed_weight`, `s2=te`.

* `rnn_num_hidden`
	
	**enc2decinit**: `s1=h`. For `weight` parameters, `s2=h` .
	
	**hidden**: `s1=h`. For `weight` parameters, `s2=2h` .
	
	**decoder\_lx**: For lstm, `s1=4h`; while for gru, `s1=3h`. For `weight` parameters, `s2=h`, except for `decoder_rnn_l0_i2h_weight`, where `s2=h+se`.

	**birnn**: For lstm, `s1=2h`; while for gru, `s1=3h/2`. For `h2h_weight` parameters, `s2=h/2`.
	
	**encoder\_lx**: For lstm, `s1=4h`; while for gru, `s1=3h`. For `weight` parameters, `s2=h`.
	
	**io**: For `target_output_weight` parameters, `s2=h`.
	
	
	
### 4. Parameters w.r.t. Hyper-parameters

From previous section, we can get the equation for calculating the number of parameters based on hyper-parameter settings.

Suppose `bpe_symbols=sb:tb`, `num_layers=sn:tn`, `num_embed=se:te`, `rnn_num_hidden=h`.

**enc2decinit**

```
decoder_rnn_enc2decinit_x_bias: (h,), 
decoder_rnn_enc2decinit_x_weight: (h, h)

where x=0,...,2*tn-1 for lstm; x=0,...,n-1 for gru.
```
The total number of `enc2decinit` parameters can be calculated as follows:

```
if rnn_cell_type == lstm:
	nparam_enc2decinit = 2*tn*h(h+1)
elif rnn_cell_type == gru:
	nparam_enc2decinit = tn*h(h+1)
```

**hidden**

```
decoder_rnn_hidden_bias: (h,), 
decoder_rnn_hidden_weight: (h, 2h)
```
The total number of `hidden` parameters can be calculated as follows:

```
nparam_hidden = h(2h+1)
```

**decoder\_lx**

``` 
decoder_rnn_lx_h2h_bias: (y,), 
decoder_rnn_lx_h2h_weight: (y, h), 
decoder_rnn_lx_i2h_bias: (y,), 
decoder_rnn_lx_i2h_weight: (y, z)

where x=0,...,tn-1. 
	  y=4h for lstm; y=3h for gru. 
	  z=h+se for x=0; z=h for x!=0.
```
The total number of `decode_lx` parameters can be calculated as follows:

```
if rnn_cell_type == lstm:
	nparam_decoder_lx = 4h(se+2*tn*(h+1))
elif rnn_cell_type == gru:
	nparam_decoder_lx = 3h(se+2*tn*(h+1))
```

**birnn**

```
encoder_birnn_forward_l0_h2h_bias: (y,), 
encoder_birnn_forward_l0_h2h_weight: (y, h/2), 
encoder_birnn_forward_l0_i2h_bias: (y,), 
encoder_birnn_forward_l0_i2h_weight: (y, se), 
encoder_birnn_reverse_l0_h2h_bias: (y,), 
encoder_birnn_reverse_l0_h2h_weight: (y, h/2), 
encoder_birnn_reverse_l0_i2h_bias: (y,), 
encoder_birnn_reverse_l0_i2h_weight: (y, se) 

where y=2h for lstm; y=3/2h for gru.
```
The total number of `birnn` parameters can be calculated as follows:

```
if rnn_cell_type == lstm:
	nparam_birnn = 2h(4+h+2*se)
elif rnn_cell_type == gru:
	nparam_birnn = 3/2h(4+h+2*se)
```

**encoder\_lx**

```
encoder_rnn_lx_h2h_bias: (y,), 
encoder_rnn_lx_h2h_weight: (y, h), 
encoder_rnn_lx_i2h_bias: (y,), 
encoder_rnn_lx_i2h_weight: (y, h)

where x=0,...,sn-2 (encoder_lx parameters do not exist for sn=1).
	  y=4h for lsrm; y=3h for gru.
```
The total number of `encoder_lx` parameters can be calculated as follows:

```
if rnn_cell_type == lstm:
	nparam_encoder_lx = 4h(sn-1)(2+2h)
elif rnn_cell_type == gru:
	nparam_encoder_lx = 3h(sn-1)(2+2h)
```
**io**

```
source_embed_weight: (ib, se), 
target_embed_weight: (ob, te), 
target_output_bias: (ob,), 
target_output_weight: (ob, h)
```

We now can get the total number of all the parameters:

```
nparam = nparam_enc2decinit + nparam_hidden + nparam_decoder_lx + nparam_birnn + nparam_encoder_lx + nparam_io

if rnn_cell_type == lstm:
	nparam = h*(-4*h+8*se+(8*sn+10*tn)(1+h)+1)+(ib*se+ob*(1+te+h))
elif rnn_cell_type == gru:
	nparam = h*(-2.5h+6*se+(6*sn+7*tn)(1+h)+1)+(ib*se+ob*(1+te+h))
```

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

The parameters (`parameter name: shape`) for a RNN model trained with `bpe_symbols=50000:50000`, `rnn_cell_type=lstm`, `num_embed=512:512`, `rnn_num_hidden=512`, `num_layers=2:2` are shown below.

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

Suppose `bpe_symbols=sb:tb`, `num_layers=sn:tn`, `num_embed=se:te`, `rnn_num_hidden=h`, where `s` stands for source/encoder, `t` stands for target/decoder. And `s1`, `s2` are the length of the first and second dimension of the parameter matrix.

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

The total number of `io` parameters can be calculated as follows:

```
nparam_io = ib*se+ob*(1+te+h)
```

We now can get the total number of all the parameters for an RNN model:

```
nparam = nparam_enc2decinit + nparam_hidden + nparam_decoder_lx + nparam_birnn + nparam_encoder_lx + nparam_io

if rnn_cell_type == lstm:
	nparam = h*(-4*h+8*se+(8*sn+10*tn)(1+h)+1)+(ib*se+ob*(1+te+h))
elif rnn_cell_type == gru:
	nparam = h*(-2.5*h+6*se+(6*sn+7*tn)(1+h)+1)+(ib*se+ob*(1+te+h))
```

## Transformer
### 1. Hyper-parameters
* `bpe_symbols`: 
* `num_layers`: Number of layers for encoder and decoder.
* `num_embed`: Embedding size for source and target tokens. 
* `transformer_feed_forward_num_hidden`: Number of hidden units in transformers feed forward layers.

### 2. Parameters
The parameters (`parameter name: shape`) for a Transformer model trained with `bpe_symbols=30000:30000`, `num_layers=1:1`, `num_embed=512`,`transformer_feed_forward_num_hidden=300` are shown below.

**decoder\_att**

```
decoder_transformer_0_att_enc_h2o_weight: (512, 512), 
decoder_transformer_0_att_enc_k2h_weight: (512, 512), 
decoder_transformer_0_att_enc_pre_norm_beta: (512,), 
decoder_transformer_0_att_enc_pre_norm_gamma: (512,), 
decoder_transformer_0_att_enc_q2h_weight: (512, 512), 
decoder_transformer_0_att_enc_v2h_weight: (512, 512),
decoder_transformer_0_att_self_h2o_weight: (512, 512), 
decoder_transformer_0_att_self_i2h_weight: (1536, 512), 
decoder_transformer_0_att_self_pre_norm_beta: (512,), 
decoder_transformer_0_att_self_pre_norm_gamma: (512,)
```

**decoder\_ff**

```
decoder_transformer_0_ff_h2o_bias: (512,), 
decoder_transformer_0_ff_h2o_weight: (512, 300), 
decoder_transformer_0_ff_i2h_bias: (300,), 
decoder_transformer_0_ff_i2h_weight: (300, 512), 
decoder_transformer_0_ff_pre_norm_beta: (512,), 
decoder_transformer_0_ff_pre_norm_gamma: (512,)
```

**decoder\_final**

```
decoder_transformer_final_process_norm_beta: (512,), 
decoder_transformer_final_process_norm_gamma: (512,)
```
**encoder\_att**

```
encoder_transformer_0_att_self_h2o_weight: (512, 512), 
encoder_transformer_0_att_self_i2h_weight: (1536, 512), 
encoder_transformer_0_att_self_pre_norm_beta: (512,), 
encoder_transformer_0_att_self_pre_norm_gamma: (512,)
```

**encoder\_ff**

```
encoder_transformer_0_ff_h2o_bias: (512,), 
encoder_transformer_0_ff_h2o_weight: (512, 300), 
encoder_transformer_0_ff_i2h_bias: (300,), 
encoder_transformer_0_ff_i2h_weight: (300, 512), 
encoder_transformer_0_ff_pre_norm_beta: (512,), 
encoder_transformer_0_ff_pre_norm_gamma: (512,)
```

**encoder\_final**

```
encoder_transformer_final_process_norm_beta: (512,),
encoder_transformer_final_process_norm_gamma: (512,)
```

**io**

```
source_embed_weight: (29624, 512), 
target_embed_weight: (28059, 512), 
target_output_bias: (28059,), 
target_output_weight: (28059, 512)
```

For the model above, the total number of parameters is 49181083.

### 3. Influence of Hyper-parameters on Parameters

Now let's see how the changes on each hyper-parameter reflect on the shape and number of parameter matrices.

Suppose `bpe_symbols=sb:tb`, `num_layers=sn:tn`, `num_embed=e`, `transformer_feed_forward_num_hidden=f`. And `s1`, `s2` are the length of the first and second dimension of the parameter matrix.

* `bpe_symbols`:
	
	?
	
* `num_layers`:

	**decoder\_att**, **decoder\_ff**, **encoder\_att**, **encoder\_ff**: `..._transformer_x_...`, where `x=0,...,n-1`.
	
* `num_embed`:

	**all**. Please see section 4 below for more details.
	
* `transformer_feed_forward_num_hidden`:

	**decoder\_ff**, **encoder\_ff**: For `..._i2h_...` matrices, `s1=f`; for `..._h2o_weight` matrices, `s2=f`.
	

### 4. Parameters w.r.t. Hyper-parameters

From previous section, we can get the equation for calculating the number of parameters based on hyper-parameter settings.

Suppose `bpe_symbols=sb:tb`, `num_layers=sn:tn`, `num_embed=e`, `transformer_feed_forward_num_hidden=f`.

**decoder\_att**

```
decoder_transformer_x_att_enc_h2o_weight: (e, e), 
decoder_transformer_x_att_enc_k2h_weight: (e, e), 
decoder_transformer_x_att_enc_pre_norm_beta: (e,), 
decoder_transformer_x_att_enc_pre_norm_gamma: (e,), 
decoder_transformer_x_att_enc_q2h_weight: (e, e), 
decoder_transformer_x_att_enc_v2h_weight: (e, e),
decoder_transformer_x_att_self_h2o_weight: (e, e), 
decoder_transformer_x_att_self_i2h_weight: (3*e, e), 
decoder_transformer_x_att_self_pre_norm_beta: (e,), 
decoder_transformer_x_att_self_pre_norm_gamma: (e,)

where x=0,...,n-1.
```
The total number of `decoder_att` parameters can be calculated as follows:

```
nparam_decoder_att = n*4e*(2e+1)
```

**decoder\_ff**

```
decoder_transformer_x_ff_h2o_bias: (e,), 
decoder_transformer_x_ff_h2o_weight: (e, f), 
decoder_transformer_x_ff_i2h_bias: (f,), 
decoder_transformer_x_ff_i2h_weight: (f, e), 
decoder_transformer_x_ff_pre_norm_beta: (e,), 
decoder_transformer_x_ff_pre_norm_gamma: (e,)

where x=0,...,n-1.
```

The total number of `decoder_ff` parameters can be calculated as follows:

```
nparam_decoder_ff = n*(2ef+3e+f)
```

**decoder\_final**

```
decoder_transformer_final_process_norm_beta: (e,), 
decoder_transformer_final_process_norm_gamma: (e,)
```

The total number of `decoder_final` parameters can be calculated as follows:

```
nparam_decoder_final = 2e
```

**encoder\_att**

```
encoder_transformer_x_att_self_h2o_weight: (e, e), 
encoder_transformer_x_att_self_i2h_weight: (3*e, e), 
encoder_transformer_x_att_self_pre_norm_beta: (e,), 
encoder_transformer_x_att_self_pre_norm_gamma: (e,)

where x=0,...,n-1.
```

The total number of `encoder_att` parameters can be calculated as follows:

```
nparam_encoder_att = n*2e*(2e+1)
```

**encoder\_ff**

```
encoder_transformer_x_ff_h2o_bias: (e,), 
encoder_transformer_x_ff_h2o_weight: (e, f), 
encoder_transformer_x_ff_i2h_bias: (f,), 
encoder_transformer_x_ff_i2h_weight: (f, e), 
encoder_transformer_x_ff_pre_norm_beta: (e,), 
encoder_transformer_x_ff_pre_norm_gamma: (e,)

where x=0,...,n-1.
```

The total number of `encoder_ff` parameters can be calculated as follows:

```
nparam_encoder_ff = n*(2ef+3e+f)
```

**encoder\_final**

```
encoder_transformer_final_process_norm_beta: (e,),
encoder_transformer_final_process_norm_gamma: (e,)
```

The total number of `encoder_final` parameters can be calculated as follows:

```
nparam_encoder_final = 2e
```

**io**

```
source_embed_weight: (ib, e), 
target_embed_weight: (ob, e), 
target_output_bias: (ob,), 
target_output_weight: (ob, e)
```

The total number of `io` parameters can be calculated as follows:

```
nparam_io = ib*e+ob*(2e+1)
```

We now can get the total number of all the parameters for a Transformer model:

```
nparam = nparam_decoder_att + nparam_decoder_ff + nparam_decoder_final + nparam_encoder_att + nparam_encoder_ff + nparam_encoder_final + nparam_io

nparam = n*(12e*e+12e+4ef+2f)+4e+(ib*e+ob*(2e+1))
```
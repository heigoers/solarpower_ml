??0
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02unknown8??/
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv_lst_m2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameconv_lst_m2d/kernel
?
'conv_lst_m2d/kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d/kernel*'
_output_shapes
:?*
dtype0
?
conv_lst_m2d/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*.
shared_nameconv_lst_m2d/recurrent_kernel
?
1conv_lst_m2d/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d/recurrent_kernel*'
_output_shapes
:@?*
dtype0
{
conv_lst_m2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameconv_lst_m2d/bias
t
%conv_lst_m2d/bias/Read/ReadVariableOpReadVariableOpconv_lst_m2d/bias*
_output_shapes	
:?*
dtype0
?
conv_lst_m2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*&
shared_nameconv_lst_m2d_1/kernel
?
)conv_lst_m2d_1/kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d_1/kernel*'
_output_shapes
:@?*
dtype0
?
conv_lst_m2d_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*0
shared_name!conv_lst_m2d_1/recurrent_kernel
?
3conv_lst_m2d_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d_1/recurrent_kernel*'
_output_shapes
:@?*
dtype0

conv_lst_m2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameconv_lst_m2d_1/bias
x
'conv_lst_m2d_1/bias/Read/ReadVariableOpReadVariableOpconv_lst_m2d_1/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv_lst_m2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d/kernel/m
?
.Adam/conv_lst_m2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d/kernel/m*'
_output_shapes
:?*
dtype0
?
$Adam/conv_lst_m2d/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*5
shared_name&$Adam/conv_lst_m2d/recurrent_kernel/m
?
8Adam/conv_lst_m2d/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/conv_lst_m2d/recurrent_kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/conv_lst_m2d/bias/m
?
,Adam/conv_lst_m2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv_lst_m2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*-
shared_nameAdam/conv_lst_m2d_1/kernel/m
?
0Adam/conv_lst_m2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_1/kernel/m*'
_output_shapes
:@?*
dtype0
?
&Adam/conv_lst_m2d_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*7
shared_name(&Adam/conv_lst_m2d_1/recurrent_kernel/m
?
:Adam/conv_lst_m2d_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/conv_lst_m2d_1/recurrent_kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d_1/bias/m
?
.Adam/conv_lst_m2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv_lst_m2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d/kernel/v
?
.Adam/conv_lst_m2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d/kernel/v*'
_output_shapes
:?*
dtype0
?
$Adam/conv_lst_m2d/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*5
shared_name&$Adam/conv_lst_m2d/recurrent_kernel/v
?
8Adam/conv_lst_m2d/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/conv_lst_m2d/recurrent_kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/conv_lst_m2d/bias/v
?
,Adam/conv_lst_m2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv_lst_m2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*-
shared_nameAdam/conv_lst_m2d_1/kernel/v
?
0Adam/conv_lst_m2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_1/kernel/v*'
_output_shapes
:@?*
dtype0
?
&Adam/conv_lst_m2d_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*7
shared_name(&Adam/conv_lst_m2d_1/recurrent_kernel/v
?
:Adam/conv_lst_m2d_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/conv_lst_m2d_1/recurrent_kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d_1/bias/v
?
.Adam/conv_lst_m2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_1/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
 
l

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemImJmKmLmM mNvOvPvQvRvS vT
 
*
0
1
2
3
4
 5
*
0
1
2
3
4
 5
?
!layer_regularization_losses
regularization_losses

"layers
#metrics
$layer_metrics
	variables
trainable_variables
%non_trainable_variables
 
~

kernel
recurrent_kernel
bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
 
 

0
1
2

0
1
2
?
*layer_regularization_losses
regularization_losses

+layers
,metrics
-layer_metrics

.states
	variables
trainable_variables
/non_trainable_variables
~

kernel
recurrent_kernel
 bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
 
 

0
1
 2

0
1
 2
?
4layer_regularization_losses
regularization_losses

5layers
6metrics
7layer_metrics

8states
	variables
trainable_variables
9non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEconv_lst_m2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv_lst_m2d/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv_lst_m2d/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv_lst_m2d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_lst_m2d_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEconv_lst_m2d_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

:0
 
 
 

0
1
2

0
1
2
?
;layer_regularization_losses
&regularization_losses

<layers
=metrics
>layer_metrics
'	variables
(trainable_variables
?non_trainable_variables
 


0
 
 
 
 
 

0
1
 2

0
1
 2
?
@layer_regularization_losses
0regularization_losses

Alayers
Bmetrics
Clayer_metrics
1	variables
2trainable_variables
Dnon_trainable_variables
 

0
 
 
 
 
4
	Etotal
	Fcount
G	variables
H	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

G	variables
rp
VARIABLE_VALUEAdam/conv_lst_m2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/conv_lst_m2d/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv_lst_m2d/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv_lst_m2d_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/conv_lst_m2d_1/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/conv_lst_m2d_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/conv_lst_m2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/conv_lst_m2d/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv_lst_m2d/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv_lst_m2d_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/conv_lst_m2d_1/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/conv_lst_m2d_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*>
_output_shapes,
*:(????????????????????*
dtype0*3
shape*:(????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_lst_m2d/kernelconv_lst_m2d/recurrent_kernelconv_lst_m2d/biasconv_lst_m2d_1/kernelconv_lst_m2d_1/recurrent_kernelconv_lst_m2d_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_27673
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'conv_lst_m2d/kernel/Read/ReadVariableOp1conv_lst_m2d/recurrent_kernel/Read/ReadVariableOp%conv_lst_m2d/bias/Read/ReadVariableOp)conv_lst_m2d_1/kernel/Read/ReadVariableOp3conv_lst_m2d_1/recurrent_kernel/Read/ReadVariableOp'conv_lst_m2d_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/conv_lst_m2d/kernel/m/Read/ReadVariableOp8Adam/conv_lst_m2d/recurrent_kernel/m/Read/ReadVariableOp,Adam/conv_lst_m2d/bias/m/Read/ReadVariableOp0Adam/conv_lst_m2d_1/kernel/m/Read/ReadVariableOp:Adam/conv_lst_m2d_1/recurrent_kernel/m/Read/ReadVariableOp.Adam/conv_lst_m2d_1/bias/m/Read/ReadVariableOp.Adam/conv_lst_m2d/kernel/v/Read/ReadVariableOp8Adam/conv_lst_m2d/recurrent_kernel/v/Read/ReadVariableOp,Adam/conv_lst_m2d/bias/v/Read/ReadVariableOp0Adam/conv_lst_m2d_1/kernel/v/Read/ReadVariableOp:Adam/conv_lst_m2d_1/recurrent_kernel/v/Read/ReadVariableOp.Adam/conv_lst_m2d_1/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_30905
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv_lst_m2d/kernelconv_lst_m2d/recurrent_kernelconv_lst_m2d/biasconv_lst_m2d_1/kernelconv_lst_m2d_1/recurrent_kernelconv_lst_m2d_1/biastotalcountAdam/conv_lst_m2d/kernel/m$Adam/conv_lst_m2d/recurrent_kernel/mAdam/conv_lst_m2d/bias/mAdam/conv_lst_m2d_1/kernel/m&Adam/conv_lst_m2d_1/recurrent_kernel/mAdam/conv_lst_m2d_1/bias/mAdam/conv_lst_m2d/kernel/v$Adam/conv_lst_m2d/recurrent_kernel/vAdam/conv_lst_m2d/bias/vAdam/conv_lst_m2d_1/kernel/v&Adam/conv_lst_m2d_1/recurrent_kernel/vAdam/conv_lst_m2d_1/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_30990??.
?
?
.__inference_conv_lstm_cell_layer_call_fn_30456

inputs
states_0
states_1"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_253762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?f
?
while_body_26678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_26278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_26302_0:@?(
while_26304_0:@?
while_26306_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_26302:@?&
while_26304:@?
while_26306:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_26302_0while_26304_0while_26306_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_262142
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5z

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"
while_26302while_26302_0"
while_26304while_26304_0"
while_26306while_26306_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
??
?
conv_lst_m2d_1_while_body_28019:
6conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter@
<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations$
 conv_lst_m2d_1_while_placeholder&
"conv_lst_m2d_1_while_placeholder_1&
"conv_lst_m2d_1_while_placeholder_2&
"conv_lst_m2d_1_while_placeholder_37
3conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice_0u
qconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0O
4conv_lst_m2d_1_while_split_readvariableop_resource_0:@?Q
6conv_lst_m2d_1_while_split_1_readvariableop_resource_0:@?E
6conv_lst_m2d_1_while_split_2_readvariableop_resource_0:	?!
conv_lst_m2d_1_while_identity#
conv_lst_m2d_1_while_identity_1#
conv_lst_m2d_1_while_identity_2#
conv_lst_m2d_1_while_identity_3#
conv_lst_m2d_1_while_identity_4#
conv_lst_m2d_1_while_identity_55
1conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slices
oconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensorM
2conv_lst_m2d_1_while_split_readvariableop_resource:@?O
4conv_lst_m2d_1_while_split_1_readvariableop_resource:@?C
4conv_lst_m2d_1_while_split_2_readvariableop_resource:	???)conv_lst_m2d_1/while/split/ReadVariableOp?+conv_lst_m2d_1/while/split_1/ReadVariableOp?+conv_lst_m2d_1/while/split_2/ReadVariableOp?
Fconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2H
Fconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0 conv_lst_m2d_1_while_placeholderOconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02:
8conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem?
$conv_lst_m2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_1/while/split/split_dim?
)conv_lst_m2d_1/while/split/ReadVariableOpReadVariableOp4conv_lst_m2d_1_while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02+
)conv_lst_m2d_1/while/split/ReadVariableOp?
conv_lst_m2d_1/while/splitSplit-conv_lst_m2d_1/while/split/split_dim:output:01conv_lst_m2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/while/split?
&conv_lst_m2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&conv_lst_m2d_1/while/split_1/split_dim?
+conv_lst_m2d_1/while/split_1/ReadVariableOpReadVariableOp6conv_lst_m2d_1_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02-
+conv_lst_m2d_1/while/split_1/ReadVariableOp?
conv_lst_m2d_1/while/split_1Split/conv_lst_m2d_1/while/split_1/split_dim:output:03conv_lst_m2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/while/split_1?
&conv_lst_m2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&conv_lst_m2d_1/while/split_2/split_dim?
+conv_lst_m2d_1/while/split_2/ReadVariableOpReadVariableOp6conv_lst_m2d_1_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02-
+conv_lst_m2d_1/while/split_2/ReadVariableOp?
conv_lst_m2d_1/while/split_2Split/conv_lst_m2d_1/while/split_2/split_dim:output:03conv_lst_m2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_1/while/split_2?
 conv_lst_m2d_1/while/convolutionConv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d_1/while/convolution?
conv_lst_m2d_1/while/BiasAddBiasAdd)conv_lst_m2d_1/while/convolution:output:0%conv_lst_m2d_1/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/BiasAdd?
"conv_lst_m2d_1/while/convolution_1Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_1?
conv_lst_m2d_1/while/BiasAdd_1BiasAdd+conv_lst_m2d_1/while/convolution_1:output:0%conv_lst_m2d_1/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_1?
"conv_lst_m2d_1/while/convolution_2Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_2?
conv_lst_m2d_1/while/BiasAdd_2BiasAdd+conv_lst_m2d_1/while/convolution_2:output:0%conv_lst_m2d_1/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_2?
"conv_lst_m2d_1/while/convolution_3Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_3?
conv_lst_m2d_1/while/BiasAdd_3BiasAdd+conv_lst_m2d_1/while/convolution_3:output:0%conv_lst_m2d_1/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_3?
"conv_lst_m2d_1/while/convolution_4Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_4?
"conv_lst_m2d_1/while/convolution_5Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_5?
"conv_lst_m2d_1/while/convolution_6Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_6?
"conv_lst_m2d_1/while/convolution_7Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_7?
conv_lst_m2d_1/while/addAddV2%conv_lst_m2d_1/while/BiasAdd:output:0+conv_lst_m2d_1/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add}
conv_lst_m2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const?
conv_lst_m2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_1?
conv_lst_m2d_1/while/MulMulconv_lst_m2d_1/while/add:z:0#conv_lst_m2d_1/while/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul?
conv_lst_m2d_1/while/Add_1AddV2conv_lst_m2d_1/while/Mul:z:0%conv_lst_m2d_1/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_1?
,conv_lst_m2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d_1/while/clip_by_value/Minimum/y?
*conv_lst_m2d_1/while/clip_by_value/MinimumMinimumconv_lst_m2d_1/while/Add_1:z:05conv_lst_m2d_1/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d_1/while/clip_by_value/Minimum?
$conv_lst_m2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d_1/while/clip_by_value/y?
"conv_lst_m2d_1/while/clip_by_valueMaximum.conv_lst_m2d_1/while/clip_by_value/Minimum:z:0-conv_lst_m2d_1/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d_1/while/clip_by_value?
conv_lst_m2d_1/while/add_2AddV2'conv_lst_m2d_1/while/BiasAdd_1:output:0+conv_lst_m2d_1/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_2?
conv_lst_m2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const_2?
conv_lst_m2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_3?
conv_lst_m2d_1/while/Mul_1Mulconv_lst_m2d_1/while/add_2:z:0%conv_lst_m2d_1/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul_1?
conv_lst_m2d_1/while/Add_3AddV2conv_lst_m2d_1/while/Mul_1:z:0%conv_lst_m2d_1/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_3?
.conv_lst_m2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_1/while/clip_by_value_1/Minimum/y?
,conv_lst_m2d_1/while/clip_by_value_1/MinimumMinimumconv_lst_m2d_1/while/Add_3:z:07conv_lst_m2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,conv_lst_m2d_1/while/clip_by_value_1/Minimum?
&conv_lst_m2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_1/while/clip_by_value_1/y?
$conv_lst_m2d_1/while/clip_by_value_1Maximum0conv_lst_m2d_1/while/clip_by_value_1/Minimum:z:0/conv_lst_m2d_1/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/while/clip_by_value_1?
conv_lst_m2d_1/while/mul_2Mul(conv_lst_m2d_1/while/clip_by_value_1:z:0"conv_lst_m2d_1_while_placeholder_3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_2?
conv_lst_m2d_1/while/add_4AddV2'conv_lst_m2d_1/while/BiasAdd_2:output:0+conv_lst_m2d_1/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_4?
conv_lst_m2d_1/while/ReluReluconv_lst_m2d_1/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Relu?
conv_lst_m2d_1/while/mul_3Mul&conv_lst_m2d_1/while/clip_by_value:z:0'conv_lst_m2d_1/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_3?
conv_lst_m2d_1/while/add_5AddV2conv_lst_m2d_1/while/mul_2:z:0conv_lst_m2d_1/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_5?
conv_lst_m2d_1/while/add_6AddV2'conv_lst_m2d_1/while/BiasAdd_3:output:0+conv_lst_m2d_1/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_6?
conv_lst_m2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const_4?
conv_lst_m2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_5?
conv_lst_m2d_1/while/Mul_4Mulconv_lst_m2d_1/while/add_6:z:0%conv_lst_m2d_1/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul_4?
conv_lst_m2d_1/while/Add_7AddV2conv_lst_m2d_1/while/Mul_4:z:0%conv_lst_m2d_1/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_7?
.conv_lst_m2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_1/while/clip_by_value_2/Minimum/y?
,conv_lst_m2d_1/while/clip_by_value_2/MinimumMinimumconv_lst_m2d_1/while/Add_7:z:07conv_lst_m2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,conv_lst_m2d_1/while/clip_by_value_2/Minimum?
&conv_lst_m2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_1/while/clip_by_value_2/y?
$conv_lst_m2d_1/while/clip_by_value_2Maximum0conv_lst_m2d_1/while/clip_by_value_2/Minimum:z:0/conv_lst_m2d_1/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/while/clip_by_value_2?
conv_lst_m2d_1/while/Relu_1Reluconv_lst_m2d_1/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Relu_1?
conv_lst_m2d_1/while/mul_5Mul(conv_lst_m2d_1/while/clip_by_value_2:z:0)conv_lst_m2d_1/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_5?
9conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"conv_lst_m2d_1_while_placeholder_1 conv_lst_m2d_1_while_placeholderconv_lst_m2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype02;
9conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem~
conv_lst_m2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_1/while/add_8/y?
conv_lst_m2d_1/while/add_8AddV2 conv_lst_m2d_1_while_placeholder%conv_lst_m2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/add_8~
conv_lst_m2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_1/while/add_9/y?
conv_lst_m2d_1/while/add_9AddV26conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter%conv_lst_m2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/add_9?
conv_lst_m2d_1/while/IdentityIdentityconv_lst_m2d_1/while/add_9:z:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/Identity?
conv_lst_m2d_1/while/Identity_1Identity<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_1?
conv_lst_m2d_1/while/Identity_2Identityconv_lst_m2d_1/while/add_8:z:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_2?
conv_lst_m2d_1/while/Identity_3IdentityIconv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_3?
conv_lst_m2d_1/while/Identity_4Identityconv_lst_m2d_1/while/mul_5:z:0^conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2!
conv_lst_m2d_1/while/Identity_4?
conv_lst_m2d_1/while/Identity_5Identityconv_lst_m2d_1/while/add_5:z:0^conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2!
conv_lst_m2d_1/while/Identity_5?
conv_lst_m2d_1/while/NoOpNoOp*^conv_lst_m2d_1/while/split/ReadVariableOp,^conv_lst_m2d_1/while/split_1/ReadVariableOp,^conv_lst_m2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
conv_lst_m2d_1/while/NoOp"h
1conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice3conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice_0"G
conv_lst_m2d_1_while_identity&conv_lst_m2d_1/while/Identity:output:0"K
conv_lst_m2d_1_while_identity_1(conv_lst_m2d_1/while/Identity_1:output:0"K
conv_lst_m2d_1_while_identity_2(conv_lst_m2d_1/while/Identity_2:output:0"K
conv_lst_m2d_1_while_identity_3(conv_lst_m2d_1/while/Identity_3:output:0"K
conv_lst_m2d_1_while_identity_4(conv_lst_m2d_1/while/Identity_4:output:0"K
conv_lst_m2d_1_while_identity_5(conv_lst_m2d_1/while/Identity_5:output:0"n
4conv_lst_m2d_1_while_split_1_readvariableop_resource6conv_lst_m2d_1_while_split_1_readvariableop_resource_0"n
4conv_lst_m2d_1_while_split_2_readvariableop_resource6conv_lst_m2d_1_while_split_2_readvariableop_resource_0"j
2conv_lst_m2d_1_while_split_readvariableop_resource4conv_lst_m2d_1_while_split_readvariableop_resource_0"?
oconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensorqconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2V
)conv_lst_m2d_1/while/split/ReadVariableOp)conv_lst_m2d_1/while/split/ReadVariableOp2Z
+conv_lst_m2d_1/while/split_1/ReadVariableOp+conv_lst_m2d_1/while/split_1/ReadVariableOp2Z
+conv_lst_m2d_1/while/split_2/ReadVariableOp+conv_lst_m2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_29646
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_29646___redundant_placeholder03
/while_while_cond_29646___redundant_placeholder13
/while_while_cond_29646___redundant_placeholder23
/while_while_cond_29646___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_27292

inputs8
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_27166*
condR
while_cond_27165*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_27690

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?$
	unknown_2:@?$
	unknown_3:@?
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_270422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?
?
while_cond_28720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_28720___redundant_placeholder03
/while_while_cond_28720___redundant_placeholder13
/while_while_cond_28720___redundant_placeholder23
/while_while_cond_28720___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
??
?
conv_lst_m2d_1_while_body_28457:
6conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter@
<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations$
 conv_lst_m2d_1_while_placeholder&
"conv_lst_m2d_1_while_placeholder_1&
"conv_lst_m2d_1_while_placeholder_2&
"conv_lst_m2d_1_while_placeholder_37
3conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice_0u
qconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0O
4conv_lst_m2d_1_while_split_readvariableop_resource_0:@?Q
6conv_lst_m2d_1_while_split_1_readvariableop_resource_0:@?E
6conv_lst_m2d_1_while_split_2_readvariableop_resource_0:	?!
conv_lst_m2d_1_while_identity#
conv_lst_m2d_1_while_identity_1#
conv_lst_m2d_1_while_identity_2#
conv_lst_m2d_1_while_identity_3#
conv_lst_m2d_1_while_identity_4#
conv_lst_m2d_1_while_identity_55
1conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slices
oconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensorM
2conv_lst_m2d_1_while_split_readvariableop_resource:@?O
4conv_lst_m2d_1_while_split_1_readvariableop_resource:@?C
4conv_lst_m2d_1_while_split_2_readvariableop_resource:	???)conv_lst_m2d_1/while/split/ReadVariableOp?+conv_lst_m2d_1/while/split_1/ReadVariableOp?+conv_lst_m2d_1/while/split_2/ReadVariableOp?
Fconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2H
Fconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0 conv_lst_m2d_1_while_placeholderOconv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02:
8conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem?
$conv_lst_m2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_1/while/split/split_dim?
)conv_lst_m2d_1/while/split/ReadVariableOpReadVariableOp4conv_lst_m2d_1_while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02+
)conv_lst_m2d_1/while/split/ReadVariableOp?
conv_lst_m2d_1/while/splitSplit-conv_lst_m2d_1/while/split/split_dim:output:01conv_lst_m2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/while/split?
&conv_lst_m2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&conv_lst_m2d_1/while/split_1/split_dim?
+conv_lst_m2d_1/while/split_1/ReadVariableOpReadVariableOp6conv_lst_m2d_1_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02-
+conv_lst_m2d_1/while/split_1/ReadVariableOp?
conv_lst_m2d_1/while/split_1Split/conv_lst_m2d_1/while/split_1/split_dim:output:03conv_lst_m2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/while/split_1?
&conv_lst_m2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&conv_lst_m2d_1/while/split_2/split_dim?
+conv_lst_m2d_1/while/split_2/ReadVariableOpReadVariableOp6conv_lst_m2d_1_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02-
+conv_lst_m2d_1/while/split_2/ReadVariableOp?
conv_lst_m2d_1/while/split_2Split/conv_lst_m2d_1/while/split_2/split_dim:output:03conv_lst_m2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_1/while/split_2?
 conv_lst_m2d_1/while/convolutionConv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d_1/while/convolution?
conv_lst_m2d_1/while/BiasAddBiasAdd)conv_lst_m2d_1/while/convolution:output:0%conv_lst_m2d_1/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/BiasAdd?
"conv_lst_m2d_1/while/convolution_1Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_1?
conv_lst_m2d_1/while/BiasAdd_1BiasAdd+conv_lst_m2d_1/while/convolution_1:output:0%conv_lst_m2d_1/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_1?
"conv_lst_m2d_1/while/convolution_2Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_2?
conv_lst_m2d_1/while/BiasAdd_2BiasAdd+conv_lst_m2d_1/while/convolution_2:output:0%conv_lst_m2d_1/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_2?
"conv_lst_m2d_1/while/convolution_3Conv2D?conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_1/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_3?
conv_lst_m2d_1/while/BiasAdd_3BiasAdd+conv_lst_m2d_1/while/convolution_3:output:0%conv_lst_m2d_1/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/while/BiasAdd_3?
"conv_lst_m2d_1/while/convolution_4Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_4?
"conv_lst_m2d_1/while/convolution_5Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_5?
"conv_lst_m2d_1/while/convolution_6Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_6?
"conv_lst_m2d_1/while/convolution_7Conv2D"conv_lst_m2d_1_while_placeholder_2%conv_lst_m2d_1/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"conv_lst_m2d_1/while/convolution_7?
conv_lst_m2d_1/while/addAddV2%conv_lst_m2d_1/while/BiasAdd:output:0+conv_lst_m2d_1/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add}
conv_lst_m2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const?
conv_lst_m2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_1?
conv_lst_m2d_1/while/MulMulconv_lst_m2d_1/while/add:z:0#conv_lst_m2d_1/while/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul?
conv_lst_m2d_1/while/Add_1AddV2conv_lst_m2d_1/while/Mul:z:0%conv_lst_m2d_1/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_1?
,conv_lst_m2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d_1/while/clip_by_value/Minimum/y?
*conv_lst_m2d_1/while/clip_by_value/MinimumMinimumconv_lst_m2d_1/while/Add_1:z:05conv_lst_m2d_1/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d_1/while/clip_by_value/Minimum?
$conv_lst_m2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d_1/while/clip_by_value/y?
"conv_lst_m2d_1/while/clip_by_valueMaximum.conv_lst_m2d_1/while/clip_by_value/Minimum:z:0-conv_lst_m2d_1/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d_1/while/clip_by_value?
conv_lst_m2d_1/while/add_2AddV2'conv_lst_m2d_1/while/BiasAdd_1:output:0+conv_lst_m2d_1/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_2?
conv_lst_m2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const_2?
conv_lst_m2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_3?
conv_lst_m2d_1/while/Mul_1Mulconv_lst_m2d_1/while/add_2:z:0%conv_lst_m2d_1/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul_1?
conv_lst_m2d_1/while/Add_3AddV2conv_lst_m2d_1/while/Mul_1:z:0%conv_lst_m2d_1/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_3?
.conv_lst_m2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_1/while/clip_by_value_1/Minimum/y?
,conv_lst_m2d_1/while/clip_by_value_1/MinimumMinimumconv_lst_m2d_1/while/Add_3:z:07conv_lst_m2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,conv_lst_m2d_1/while/clip_by_value_1/Minimum?
&conv_lst_m2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_1/while/clip_by_value_1/y?
$conv_lst_m2d_1/while/clip_by_value_1Maximum0conv_lst_m2d_1/while/clip_by_value_1/Minimum:z:0/conv_lst_m2d_1/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/while/clip_by_value_1?
conv_lst_m2d_1/while/mul_2Mul(conv_lst_m2d_1/while/clip_by_value_1:z:0"conv_lst_m2d_1_while_placeholder_3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_2?
conv_lst_m2d_1/while/add_4AddV2'conv_lst_m2d_1/while/BiasAdd_2:output:0+conv_lst_m2d_1/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_4?
conv_lst_m2d_1/while/ReluReluconv_lst_m2d_1/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Relu?
conv_lst_m2d_1/while/mul_3Mul&conv_lst_m2d_1/while/clip_by_value:z:0'conv_lst_m2d_1/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_3?
conv_lst_m2d_1/while/add_5AddV2conv_lst_m2d_1/while/mul_2:z:0conv_lst_m2d_1/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_5?
conv_lst_m2d_1/while/add_6AddV2'conv_lst_m2d_1/while/BiasAdd_3:output:0+conv_lst_m2d_1/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/add_6?
conv_lst_m2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/while/Const_4?
conv_lst_m2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/while/Const_5?
conv_lst_m2d_1/while/Mul_4Mulconv_lst_m2d_1/while/add_6:z:0%conv_lst_m2d_1/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Mul_4?
conv_lst_m2d_1/while/Add_7AddV2conv_lst_m2d_1/while/Mul_4:z:0%conv_lst_m2d_1/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Add_7?
.conv_lst_m2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_1/while/clip_by_value_2/Minimum/y?
,conv_lst_m2d_1/while/clip_by_value_2/MinimumMinimumconv_lst_m2d_1/while/Add_7:z:07conv_lst_m2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,conv_lst_m2d_1/while/clip_by_value_2/Minimum?
&conv_lst_m2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_1/while/clip_by_value_2/y?
$conv_lst_m2d_1/while/clip_by_value_2Maximum0conv_lst_m2d_1/while/clip_by_value_2/Minimum:z:0/conv_lst_m2d_1/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/while/clip_by_value_2?
conv_lst_m2d_1/while/Relu_1Reluconv_lst_m2d_1/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/Relu_1?
conv_lst_m2d_1/while/mul_5Mul(conv_lst_m2d_1/while/clip_by_value_2:z:0)conv_lst_m2d_1/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/while/mul_5?
9conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"conv_lst_m2d_1_while_placeholder_1 conv_lst_m2d_1_while_placeholderconv_lst_m2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype02;
9conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem~
conv_lst_m2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_1/while/add_8/y?
conv_lst_m2d_1/while/add_8AddV2 conv_lst_m2d_1_while_placeholder%conv_lst_m2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/add_8~
conv_lst_m2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_1/while/add_9/y?
conv_lst_m2d_1/while/add_9AddV26conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter%conv_lst_m2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/add_9?
conv_lst_m2d_1/while/IdentityIdentityconv_lst_m2d_1/while/add_9:z:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/Identity?
conv_lst_m2d_1/while/Identity_1Identity<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_1?
conv_lst_m2d_1/while/Identity_2Identityconv_lst_m2d_1/while/add_8:z:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_2?
conv_lst_m2d_1/while/Identity_3IdentityIconv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_1/while/Identity_3?
conv_lst_m2d_1/while/Identity_4Identityconv_lst_m2d_1/while/mul_5:z:0^conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2!
conv_lst_m2d_1/while/Identity_4?
conv_lst_m2d_1/while/Identity_5Identityconv_lst_m2d_1/while/add_5:z:0^conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2!
conv_lst_m2d_1/while/Identity_5?
conv_lst_m2d_1/while/NoOpNoOp*^conv_lst_m2d_1/while/split/ReadVariableOp,^conv_lst_m2d_1/while/split_1/ReadVariableOp,^conv_lst_m2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
conv_lst_m2d_1/while/NoOp"h
1conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice3conv_lst_m2d_1_while_conv_lst_m2d_1_strided_slice_0"G
conv_lst_m2d_1_while_identity&conv_lst_m2d_1/while/Identity:output:0"K
conv_lst_m2d_1_while_identity_1(conv_lst_m2d_1/while/Identity_1:output:0"K
conv_lst_m2d_1_while_identity_2(conv_lst_m2d_1/while/Identity_2:output:0"K
conv_lst_m2d_1_while_identity_3(conv_lst_m2d_1/while/Identity_3:output:0"K
conv_lst_m2d_1_while_identity_4(conv_lst_m2d_1/while/Identity_4:output:0"K
conv_lst_m2d_1_while_identity_5(conv_lst_m2d_1/while/Identity_5:output:0"n
4conv_lst_m2d_1_while_split_1_readvariableop_resource6conv_lst_m2d_1_while_split_1_readvariableop_resource_0"n
4conv_lst_m2d_1_while_split_2_readvariableop_resource6conv_lst_m2d_1_while_split_2_readvariableop_resource_0"j
2conv_lst_m2d_1_while_split_readvariableop_resource4conv_lst_m2d_1_while_split_readvariableop_resource_0"?
oconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensorqconv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2V
)conv_lst_m2d_1/while/split/ReadVariableOp)conv_lst_m2d_1/while/split/ReadVariableOp2Z
+conv_lst_m2d_1/while/split_1/ReadVariableOp+conv_lst_m2d_1/while/split_1/ReadVariableOp2Z
+conv_lst_m2d_1/while/split_2/ReadVariableOp+conv_lst_m2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?9
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_26108

inputs"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_260262
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26040*
condR
while_cond_26039*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityp
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?f
?
while_body_29381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29773
inputs_08
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilex

zeros_like	ZerosLikeinputs_0*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_29647*
condR
while_cond_29646*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:h d
>
_output_shapes,
*:(????????????????????@
"
_user_specified_name
inputs/0
?
?
@__inference_model_layer_call_and_return_conditional_losses_27580

inputs-
conv_lst_m2d_27565:?-
conv_lst_m2d_27567:@?!
conv_lst_m2d_27569:	?/
conv_lst_m2d_1_27572:@?/
conv_lst_m2d_1_27574:@?#
conv_lst_m2d_1_27576:	?
identity??$conv_lst_m2d/StatefulPartitionedCall?&conv_lst_m2d_1/StatefulPartitionedCall?
$conv_lst_m2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lst_m2d_27565conv_lst_m2d_27567conv_lst_m2d_27569*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_275342&
$conv_lst_m2d/StatefulPartitionedCall?
&conv_lst_m2d_1/StatefulPartitionedCallStatefulPartitionedCall-conv_lst_m2d/StatefulPartitionedCall:output:0conv_lst_m2d_1_27572conv_lst_m2d_1_27574conv_lst_m2d_1_27576*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_272922(
&conv_lst_m2d_1/StatefulPartitionedCall?
IdentityIdentity/conv_lst_m2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp%^conv_lst_m2d/StatefulPartitionedCall'^conv_lst_m2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2L
$conv_lst_m2d/StatefulPartitionedCall$conv_lst_m2d/StatefulPartitionedCall2P
&conv_lst_m2d_1/StatefulPartitionedCall&conv_lst_m2d_1/StatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_27630
input_1-
conv_lst_m2d_27615:?-
conv_lst_m2d_27617:@?!
conv_lst_m2d_27619:	?/
conv_lst_m2d_1_27622:@?/
conv_lst_m2d_1_27624:@?#
conv_lst_m2d_1_27626:	?
identity??$conv_lst_m2d/StatefulPartitionedCall?&conv_lst_m2d_1/StatefulPartitionedCall?
$conv_lst_m2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_lst_m2d_27615conv_lst_m2d_27617conv_lst_m2d_27619*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_268042&
$conv_lst_m2d/StatefulPartitionedCall?
&conv_lst_m2d_1/StatefulPartitionedCallStatefulPartitionedCall-conv_lst_m2d/StatefulPartitionedCall:output:0conv_lst_m2d_1_27622conv_lst_m2d_1_27624conv_lst_m2d_1_27626*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_270332(
&conv_lst_m2d_1/StatefulPartitionedCall?
IdentityIdentity/conv_lst_m2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp%^conv_lst_m2d/StatefulPartitionedCall'^conv_lst_m2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2L
$conv_lst_m2d/StatefulPartitionedCall$conv_lst_m2d/StatefulPartitionedCall2P
&conv_lst_m2d_1/StatefulPartitionedCall&conv_lst_m2d_1/StatefulPartitionedCall:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1
?	
?
.__inference_conv_lst_m2d_1_layer_call_fn_29540

inputs"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_270332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?f
?
while_body_26907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_26040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_26064_0:@?(
while_26066_0:@?
while_26068_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_26064:@?&
while_26066:@?
while_26068:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_26064_0while_26066_0while_26068_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_260262
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5z

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"
while_26064while_26064_0"
while_26066while_26066_0"
while_26068while_26068_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_26277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_26277___redundant_placeholder03
/while_while_cond_26277___redundant_placeholder13
/while_while_cond_26277___redundant_placeholder23
/while_while_cond_26277___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30217

inputs8
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_30091*
condR
while_cond_30090*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30439

inputs8
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_30313*
condR
while_cond_30312*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_28145

inputsE
*conv_lst_m2d_split_readvariableop_resource:?G
,conv_lst_m2d_split_1_readvariableop_resource:@?;
,conv_lst_m2d_split_2_readvariableop_resource:	?G
,conv_lst_m2d_1_split_readvariableop_resource:@?I
.conv_lst_m2d_1_split_1_readvariableop_resource:@?=
.conv_lst_m2d_1_split_2_readvariableop_resource:	?
identity??!conv_lst_m2d/split/ReadVariableOp?#conv_lst_m2d/split_1/ReadVariableOp?#conv_lst_m2d/split_2/ReadVariableOp?conv_lst_m2d/while?#conv_lst_m2d_1/split/ReadVariableOp?%conv_lst_m2d_1/split_1/ReadVariableOp?%conv_lst_m2d_1/split_2/ReadVariableOp?conv_lst_m2d_1/while?
conv_lst_m2d/zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2
conv_lst_m2d/zeros_like?
"conv_lst_m2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"conv_lst_m2d/Sum/reduction_indices?
conv_lst_m2d/SumSumconv_lst_m2d/zeros_like:y:0+conv_lst_m2d/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
conv_lst_m2d/Sum?
conv_lst_m2d/zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
conv_lst_m2d/zeros?
conv_lst_m2d/convolutionConv2Dconv_lst_m2d/Sum:output:0conv_lst_m2d/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution?
conv_lst_m2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d/transpose/perm?
conv_lst_m2d/transpose	Transposeinputs$conv_lst_m2d/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
conv_lst_m2d/transposer
conv_lst_m2d/ShapeShapeconv_lst_m2d/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d/Shape?
 conv_lst_m2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv_lst_m2d/strided_slice/stack?
"conv_lst_m2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"conv_lst_m2d/strided_slice/stack_1?
"conv_lst_m2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"conv_lst_m2d/strided_slice/stack_2?
conv_lst_m2d/strided_sliceStridedSliceconv_lst_m2d/Shape:output:0)conv_lst_m2d/strided_slice/stack:output:0+conv_lst_m2d/strided_slice/stack_1:output:0+conv_lst_m2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d/strided_slice?
(conv_lst_m2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(conv_lst_m2d/TensorArrayV2/element_shape?
conv_lst_m2d/TensorArrayV2TensorListReserve1conv_lst_m2d/TensorArrayV2/element_shape:output:0#conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d/TensorArrayV2?
Bconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2D
Bconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape?
4conv_lst_m2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d/transpose:y:0Kconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor?
"conv_lst_m2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d/strided_slice_1/stack?
$conv_lst_m2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_1/stack_1?
$conv_lst_m2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_1/stack_2?
conv_lst_m2d/strided_slice_1StridedSliceconv_lst_m2d/transpose:y:0+conv_lst_m2d/strided_slice_1/stack:output:0-conv_lst_m2d/strided_slice_1/stack_1:output:0-conv_lst_m2d/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
conv_lst_m2d/strided_slice_1~
conv_lst_m2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/split/split_dim?
!conv_lst_m2d/split/ReadVariableOpReadVariableOp*conv_lst_m2d_split_readvariableop_resource*'
_output_shapes
:?*
dtype02#
!conv_lst_m2d/split/ReadVariableOp?
conv_lst_m2d/splitSplit%conv_lst_m2d/split/split_dim:output:0)conv_lst_m2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
conv_lst_m2d/split?
conv_lst_m2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d/split_1/split_dim?
#conv_lst_m2d/split_1/ReadVariableOpReadVariableOp,conv_lst_m2d_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#conv_lst_m2d/split_1/ReadVariableOp?
conv_lst_m2d/split_1Split'conv_lst_m2d/split_1/split_dim:output:0+conv_lst_m2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d/split_1?
conv_lst_m2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_lst_m2d/split_2/split_dim?
#conv_lst_m2d/split_2/ReadVariableOpReadVariableOp,conv_lst_m2d_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#conv_lst_m2d/split_2/ReadVariableOp?
conv_lst_m2d/split_2Split'conv_lst_m2d/split_2/split_dim:output:0+conv_lst_m2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d/split_2?
conv_lst_m2d/convolution_1Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_1?
conv_lst_m2d/BiasAddBiasAdd#conv_lst_m2d/convolution_1:output:0conv_lst_m2d/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd?
conv_lst_m2d/convolution_2Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_2?
conv_lst_m2d/BiasAdd_1BiasAdd#conv_lst_m2d/convolution_2:output:0conv_lst_m2d/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_1?
conv_lst_m2d/convolution_3Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_3?
conv_lst_m2d/BiasAdd_2BiasAdd#conv_lst_m2d/convolution_3:output:0conv_lst_m2d/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_2?
conv_lst_m2d/convolution_4Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_4?
conv_lst_m2d/BiasAdd_3BiasAdd#conv_lst_m2d/convolution_4:output:0conv_lst_m2d/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_3?
conv_lst_m2d/convolution_5Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_5?
conv_lst_m2d/convolution_6Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_6?
conv_lst_m2d/convolution_7Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_7?
conv_lst_m2d/convolution_8Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_8?
conv_lst_m2d/addAddV2conv_lst_m2d/BiasAdd:output:0#conv_lst_m2d/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/addm
conv_lst_m2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Constq
conv_lst_m2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_1?
conv_lst_m2d/MulMulconv_lst_m2d/add:z:0conv_lst_m2d/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul?
conv_lst_m2d/Add_1AddV2conv_lst_m2d/Mul:z:0conv_lst_m2d/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_1?
$conv_lst_m2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$conv_lst_m2d/clip_by_value/Minimum/y?
"conv_lst_m2d/clip_by_value/MinimumMinimumconv_lst_m2d/Add_1:z:0-conv_lst_m2d/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/clip_by_value/Minimum?
conv_lst_m2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d/clip_by_value/y?
conv_lst_m2d/clip_by_valueMaximum&conv_lst_m2d/clip_by_value/Minimum:z:0%conv_lst_m2d/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value?
conv_lst_m2d/add_2AddV2conv_lst_m2d/BiasAdd_1:output:0#conv_lst_m2d/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_2q
conv_lst_m2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Const_2q
conv_lst_m2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_3?
conv_lst_m2d/Mul_1Mulconv_lst_m2d/add_2:z:0conv_lst_m2d/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul_1?
conv_lst_m2d/Add_3AddV2conv_lst_m2d/Mul_1:z:0conv_lst_m2d/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_3?
&conv_lst_m2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d/clip_by_value_1/Minimum/y?
$conv_lst_m2d/clip_by_value_1/MinimumMinimumconv_lst_m2d/Add_3:z:0/conv_lst_m2d/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d/clip_by_value_1/Minimum?
conv_lst_m2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d/clip_by_value_1/y?
conv_lst_m2d/clip_by_value_1Maximum(conv_lst_m2d/clip_by_value_1/Minimum:z:0'conv_lst_m2d/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value_1?
conv_lst_m2d/mul_2Mul conv_lst_m2d/clip_by_value_1:z:0!conv_lst_m2d/convolution:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_2?
conv_lst_m2d/add_4AddV2conv_lst_m2d/BiasAdd_2:output:0#conv_lst_m2d/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_4?
conv_lst_m2d/ReluReluconv_lst_m2d/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Relu?
conv_lst_m2d/mul_3Mulconv_lst_m2d/clip_by_value:z:0conv_lst_m2d/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_3?
conv_lst_m2d/add_5AddV2conv_lst_m2d/mul_2:z:0conv_lst_m2d/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_5?
conv_lst_m2d/add_6AddV2conv_lst_m2d/BiasAdd_3:output:0#conv_lst_m2d/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_6q
conv_lst_m2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Const_4q
conv_lst_m2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_5?
conv_lst_m2d/Mul_4Mulconv_lst_m2d/add_6:z:0conv_lst_m2d/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul_4?
conv_lst_m2d/Add_7AddV2conv_lst_m2d/Mul_4:z:0conv_lst_m2d/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_7?
&conv_lst_m2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d/clip_by_value_2/Minimum/y?
$conv_lst_m2d/clip_by_value_2/MinimumMinimumconv_lst_m2d/Add_7:z:0/conv_lst_m2d/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d/clip_by_value_2/Minimum?
conv_lst_m2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d/clip_by_value_2/y?
conv_lst_m2d/clip_by_value_2Maximum(conv_lst_m2d/clip_by_value_2/Minimum:z:0'conv_lst_m2d/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value_2?
conv_lst_m2d/Relu_1Reluconv_lst_m2d/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Relu_1?
conv_lst_m2d/mul_5Mul conv_lst_m2d/clip_by_value_2:z:0!conv_lst_m2d/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_5?
*conv_lst_m2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2,
*conv_lst_m2d/TensorArrayV2_1/element_shape?
conv_lst_m2d/TensorArrayV2_1TensorListReserve3conv_lst_m2d/TensorArrayV2_1/element_shape:output:0#conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d/TensorArrayV2_1h
conv_lst_m2d/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d/time?
%conv_lst_m2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv_lst_m2d/while/maximum_iterations?
conv_lst_m2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
conv_lst_m2d/while/loop_counter?
conv_lst_m2d/whileWhile(conv_lst_m2d/while/loop_counter:output:0.conv_lst_m2d/while/maximum_iterations:output:0conv_lst_m2d/time:output:0%conv_lst_m2d/TensorArrayV2_1:handle:0!conv_lst_m2d/convolution:output:0!conv_lst_m2d/convolution:output:0#conv_lst_m2d/strided_slice:output:0Dconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor:output_handle:0*conv_lst_m2d_split_readvariableop_resource,conv_lst_m2d_split_1_readvariableop_resource,conv_lst_m2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
conv_lst_m2d_while_body_27801*)
cond!R
conv_lst_m2d_while_cond_27800*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
conv_lst_m2d/while?
=conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2?
=conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape?
/conv_lst_m2d/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d/while:output:3Fconv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype021
/conv_lst_m2d/TensorArrayV2Stack/TensorListStack?
"conv_lst_m2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"conv_lst_m2d/strided_slice_2/stack?
$conv_lst_m2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d/strided_slice_2/stack_1?
$conv_lst_m2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_2/stack_2?
conv_lst_m2d/strided_slice_2StridedSlice8conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:0+conv_lst_m2d/strided_slice_2/stack:output:0-conv_lst_m2d/strided_slice_2/stack_1:output:0-conv_lst_m2d/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
conv_lst_m2d/strided_slice_2?
conv_lst_m2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d/transpose_1/perm?
conv_lst_m2d/transpose_1	Transpose8conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:0&conv_lst_m2d/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d/transpose_1?
conv_lst_m2d_1/zeros_like	ZerosLikeconv_lst_m2d/transpose_1:y:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/zeros_like?
$conv_lst_m2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_1/Sum/reduction_indices?
conv_lst_m2d_1/SumSumconv_lst_m2d_1/zeros_like:y:0-conv_lst_m2d_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Sum?
$conv_lst_m2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2&
$conv_lst_m2d_1/zeros/shape_as_tensor}
conv_lst_m2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d_1/zeros/Const?
conv_lst_m2d_1/zerosFill-conv_lst_m2d_1/zeros/shape_as_tensor:output:0#conv_lst_m2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:@@2
conv_lst_m2d_1/zeros?
conv_lst_m2d_1/convolutionConv2Dconv_lst_m2d_1/Sum:output:0conv_lst_m2d_1/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution?
conv_lst_m2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d_1/transpose/perm?
conv_lst_m2d_1/transpose	Transposeconv_lst_m2d/transpose_1:y:0&conv_lst_m2d_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/transposex
conv_lst_m2d_1/ShapeShapeconv_lst_m2d_1/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d_1/Shape?
"conv_lst_m2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d_1/strided_slice/stack?
$conv_lst_m2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_1/strided_slice/stack_1?
$conv_lst_m2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_1/strided_slice/stack_2?
conv_lst_m2d_1/strided_sliceStridedSliceconv_lst_m2d_1/Shape:output:0+conv_lst_m2d_1/strided_slice/stack:output:0-conv_lst_m2d_1/strided_slice/stack_1:output:0-conv_lst_m2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d_1/strided_slice?
*conv_lst_m2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*conv_lst_m2d_1/TensorArrayV2/element_shape?
conv_lst_m2d_1/TensorArrayV2TensorListReserve3conv_lst_m2d_1/TensorArrayV2/element_shape:output:0%conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d_1/TensorArrayV2?
Dconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2F
Dconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
6conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d_1/transpose:y:0Mconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor?
$conv_lst_m2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d_1/strided_slice_1/stack?
&conv_lst_m2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_1/stack_1?
&conv_lst_m2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_1/stack_2?
conv_lst_m2d_1/strided_slice_1StridedSliceconv_lst_m2d_1/transpose:y:0-conv_lst_m2d_1/strided_slice_1/stack:output:0/conv_lst_m2d_1/strided_slice_1/stack_1:output:0/conv_lst_m2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2 
conv_lst_m2d_1/strided_slice_1?
conv_lst_m2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d_1/split/split_dim?
#conv_lst_m2d_1/split/ReadVariableOpReadVariableOp,conv_lst_m2d_1_split_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#conv_lst_m2d_1/split/ReadVariableOp?
conv_lst_m2d_1/splitSplit'conv_lst_m2d_1/split/split_dim:output:0+conv_lst_m2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/split?
 conv_lst_m2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 conv_lst_m2d_1/split_1/split_dim?
%conv_lst_m2d_1/split_1/ReadVariableOpReadVariableOp.conv_lst_m2d_1_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%conv_lst_m2d_1/split_1/ReadVariableOp?
conv_lst_m2d_1/split_1Split)conv_lst_m2d_1/split_1/split_dim:output:0-conv_lst_m2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/split_1?
 conv_lst_m2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv_lst_m2d_1/split_2/split_dim?
%conv_lst_m2d_1/split_2/ReadVariableOpReadVariableOp.conv_lst_m2d_1_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%conv_lst_m2d_1/split_2/ReadVariableOp?
conv_lst_m2d_1/split_2Split)conv_lst_m2d_1/split_2/split_dim:output:0-conv_lst_m2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_1/split_2?
conv_lst_m2d_1/convolution_1Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_1?
conv_lst_m2d_1/BiasAddBiasAdd%conv_lst_m2d_1/convolution_1:output:0conv_lst_m2d_1/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd?
conv_lst_m2d_1/convolution_2Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_2?
conv_lst_m2d_1/BiasAdd_1BiasAdd%conv_lst_m2d_1/convolution_2:output:0conv_lst_m2d_1/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_1?
conv_lst_m2d_1/convolution_3Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_3?
conv_lst_m2d_1/BiasAdd_2BiasAdd%conv_lst_m2d_1/convolution_3:output:0conv_lst_m2d_1/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_2?
conv_lst_m2d_1/convolution_4Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_4?
conv_lst_m2d_1/BiasAdd_3BiasAdd%conv_lst_m2d_1/convolution_4:output:0conv_lst_m2d_1/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_3?
conv_lst_m2d_1/convolution_5Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_5?
conv_lst_m2d_1/convolution_6Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_6?
conv_lst_m2d_1/convolution_7Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_7?
conv_lst_m2d_1/convolution_8Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_8?
conv_lst_m2d_1/addAddV2conv_lst_m2d_1/BiasAdd:output:0%conv_lst_m2d_1/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/addq
conv_lst_m2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Constu
conv_lst_m2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_1?
conv_lst_m2d_1/MulMulconv_lst_m2d_1/add:z:0conv_lst_m2d_1/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul?
conv_lst_m2d_1/Add_1AddV2conv_lst_m2d_1/Mul:z:0conv_lst_m2d_1/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_1?
&conv_lst_m2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d_1/clip_by_value/Minimum/y?
$conv_lst_m2d_1/clip_by_value/MinimumMinimumconv_lst_m2d_1/Add_1:z:0/conv_lst_m2d_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/clip_by_value/Minimum?
conv_lst_m2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d_1/clip_by_value/y?
conv_lst_m2d_1/clip_by_valueMaximum(conv_lst_m2d_1/clip_by_value/Minimum:z:0'conv_lst_m2d_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/clip_by_value?
conv_lst_m2d_1/add_2AddV2!conv_lst_m2d_1/BiasAdd_1:output:0%conv_lst_m2d_1/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_2u
conv_lst_m2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Const_2u
conv_lst_m2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_3?
conv_lst_m2d_1/Mul_1Mulconv_lst_m2d_1/add_2:z:0conv_lst_m2d_1/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul_1?
conv_lst_m2d_1/Add_3AddV2conv_lst_m2d_1/Mul_1:z:0conv_lst_m2d_1/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_3?
(conv_lst_m2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_1/clip_by_value_1/Minimum/y?
&conv_lst_m2d_1/clip_by_value_1/MinimumMinimumconv_lst_m2d_1/Add_3:z:01conv_lst_m2d_1/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2(
&conv_lst_m2d_1/clip_by_value_1/Minimum?
 conv_lst_m2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_1/clip_by_value_1/y?
conv_lst_m2d_1/clip_by_value_1Maximum*conv_lst_m2d_1/clip_by_value_1/Minimum:z:0)conv_lst_m2d_1/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/clip_by_value_1?
conv_lst_m2d_1/mul_2Mul"conv_lst_m2d_1/clip_by_value_1:z:0#conv_lst_m2d_1/convolution:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_2?
conv_lst_m2d_1/add_4AddV2!conv_lst_m2d_1/BiasAdd_2:output:0%conv_lst_m2d_1/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_4?
conv_lst_m2d_1/ReluReluconv_lst_m2d_1/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Relu?
conv_lst_m2d_1/mul_3Mul conv_lst_m2d_1/clip_by_value:z:0!conv_lst_m2d_1/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_3?
conv_lst_m2d_1/add_5AddV2conv_lst_m2d_1/mul_2:z:0conv_lst_m2d_1/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_5?
conv_lst_m2d_1/add_6AddV2!conv_lst_m2d_1/BiasAdd_3:output:0%conv_lst_m2d_1/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_6u
conv_lst_m2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Const_4u
conv_lst_m2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_5?
conv_lst_m2d_1/Mul_4Mulconv_lst_m2d_1/add_6:z:0conv_lst_m2d_1/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul_4?
conv_lst_m2d_1/Add_7AddV2conv_lst_m2d_1/Mul_4:z:0conv_lst_m2d_1/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_7?
(conv_lst_m2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_1/clip_by_value_2/Minimum/y?
&conv_lst_m2d_1/clip_by_value_2/MinimumMinimumconv_lst_m2d_1/Add_7:z:01conv_lst_m2d_1/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2(
&conv_lst_m2d_1/clip_by_value_2/Minimum?
 conv_lst_m2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_1/clip_by_value_2/y?
conv_lst_m2d_1/clip_by_value_2Maximum*conv_lst_m2d_1/clip_by_value_2/Minimum:z:0)conv_lst_m2d_1/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/clip_by_value_2?
conv_lst_m2d_1/Relu_1Reluconv_lst_m2d_1/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Relu_1?
conv_lst_m2d_1/mul_5Mul"conv_lst_m2d_1/clip_by_value_2:z:0#conv_lst_m2d_1/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_5?
,conv_lst_m2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2.
,conv_lst_m2d_1/TensorArrayV2_1/element_shape?
conv_lst_m2d_1/TensorArrayV2_1TensorListReserve5conv_lst_m2d_1/TensorArrayV2_1/element_shape:output:0%conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
conv_lst_m2d_1/TensorArrayV2_1l
conv_lst_m2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d_1/time?
'conv_lst_m2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'conv_lst_m2d_1/while/maximum_iterations?
!conv_lst_m2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv_lst_m2d_1/while/loop_counter?
conv_lst_m2d_1/whileWhile*conv_lst_m2d_1/while/loop_counter:output:00conv_lst_m2d_1/while/maximum_iterations:output:0conv_lst_m2d_1/time:output:0'conv_lst_m2d_1/TensorArrayV2_1:handle:0#conv_lst_m2d_1/convolution:output:0#conv_lst_m2d_1/convolution:output:0%conv_lst_m2d_1/strided_slice:output:0Fconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0,conv_lst_m2d_1_split_readvariableop_resource.conv_lst_m2d_1_split_1_readvariableop_resource.conv_lst_m2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
conv_lst_m2d_1_while_body_28019*+
cond#R!
conv_lst_m2d_1_while_cond_28018*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
conv_lst_m2d_1/while?
?conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2A
?conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape?
1conv_lst_m2d_1/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d_1/while:output:3Hconv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype023
1conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack?
$conv_lst_m2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$conv_lst_m2d_1/strided_slice_2/stack?
&conv_lst_m2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_lst_m2d_1/strided_slice_2/stack_1?
&conv_lst_m2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_2/stack_2?
conv_lst_m2d_1/strided_slice_2StridedSlice:conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:0-conv_lst_m2d_1/strided_slice_2/stack:output:0/conv_lst_m2d_1/strided_slice_2/stack_1:output:0/conv_lst_m2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2 
conv_lst_m2d_1/strided_slice_2?
conv_lst_m2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2!
conv_lst_m2d_1/transpose_1/perm?
conv_lst_m2d_1/transpose_1	Transpose:conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:0(conv_lst_m2d_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/transpose_1?
IdentityIdentityconv_lst_m2d_1/transpose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp"^conv_lst_m2d/split/ReadVariableOp$^conv_lst_m2d/split_1/ReadVariableOp$^conv_lst_m2d/split_2/ReadVariableOp^conv_lst_m2d/while$^conv_lst_m2d_1/split/ReadVariableOp&^conv_lst_m2d_1/split_1/ReadVariableOp&^conv_lst_m2d_1/split_2/ReadVariableOp^conv_lst_m2d_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2F
!conv_lst_m2d/split/ReadVariableOp!conv_lst_m2d/split/ReadVariableOp2J
#conv_lst_m2d/split_1/ReadVariableOp#conv_lst_m2d/split_1/ReadVariableOp2J
#conv_lst_m2d/split_2/ReadVariableOp#conv_lst_m2d/split_2/ReadVariableOp2(
conv_lst_m2d/whileconv_lst_m2d/while2J
#conv_lst_m2d_1/split/ReadVariableOp#conv_lst_m2d_1/split/ReadVariableOp2N
%conv_lst_m2d_1/split_1/ReadVariableOp%conv_lst_m2d_1/split_1/ReadVariableOp2N
%conv_lst_m2d_1/split_2/ReadVariableOp%conv_lst_m2d_1/split_2/ReadVariableOp2,
conv_lst_m2d_1/whileconv_lst_m2d_1/while:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_25564

inputs

states
states_18
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates
?f
?
while_body_27166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_conv_lstm_cell_layer_call_fn_30473

inputs
states_0
states_1"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_255642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?E
?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_26214

inputs

states
states_18
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates
?f
?
while_body_27408
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?7
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_25458

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_253762
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_25390*
condR
while_cond_25389*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityp
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30623

inputs
states_0
states_18
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?
?
0__inference_conv_lstm_cell_1_layer_call_fn_30640

inputs
states_0
states_1"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_260262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?
?
conv_lst_m2d_1_while_cond_28018:
6conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter@
<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations$
 conv_lst_m2d_1_while_placeholder&
"conv_lst_m2d_1_while_placeholder_1&
"conv_lst_m2d_1_while_placeholder_2&
"conv_lst_m2d_1_while_placeholder_3:
6conv_lst_m2d_1_while_less_conv_lst_m2d_1_strided_sliceQ
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28018___redundant_placeholder0Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28018___redundant_placeholder1Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28018___redundant_placeholder2Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28018___redundant_placeholder3!
conv_lst_m2d_1_while_identity
?
conv_lst_m2d_1/while/LessLess conv_lst_m2d_1_while_placeholder6conv_lst_m2d_1_while_less_conv_lst_m2d_1_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/Less?
conv_lst_m2d_1/while/IdentityIdentityconv_lst_m2d_1/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d_1/while/Identity"G
conv_lst_m2d_1_while_identity&conv_lst_m2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?	
?
,__inference_conv_lst_m2d_layer_call_fn_28627

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_275342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_27648
input_1-
conv_lst_m2d_27633:?-
conv_lst_m2d_27635:@?!
conv_lst_m2d_27637:	?/
conv_lst_m2d_1_27640:@?/
conv_lst_m2d_1_27642:@?#
conv_lst_m2d_1_27644:	?
identity??$conv_lst_m2d/StatefulPartitionedCall?&conv_lst_m2d_1/StatefulPartitionedCall?
$conv_lst_m2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_lst_m2d_27633conv_lst_m2d_27635conv_lst_m2d_27637*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_275342&
$conv_lst_m2d/StatefulPartitionedCall?
&conv_lst_m2d_1/StatefulPartitionedCallStatefulPartitionedCall-conv_lst_m2d/StatefulPartitionedCall:output:0conv_lst_m2d_1_27640conv_lst_m2d_1_27642conv_lst_m2d_1_27644*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_272922(
&conv_lst_m2d_1/StatefulPartitionedCall?
IdentityIdentity/conv_lst_m2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp%^conv_lst_m2d/StatefulPartitionedCall'^conv_lst_m2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2L
$conv_lst_m2d/StatefulPartitionedCall$conv_lst_m2d/StatefulPartitionedCall2P
&conv_lst_m2d_1/StatefulPartitionedCall&conv_lst_m2d_1/StatefulPartitionedCall:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1
??
?
 __inference__wrapped_model_25274
input_1K
0model_conv_lst_m2d_split_readvariableop_resource:?M
2model_conv_lst_m2d_split_1_readvariableop_resource:@?A
2model_conv_lst_m2d_split_2_readvariableop_resource:	?M
2model_conv_lst_m2d_1_split_readvariableop_resource:@?O
4model_conv_lst_m2d_1_split_1_readvariableop_resource:@?C
4model_conv_lst_m2d_1_split_2_readvariableop_resource:	?
identity??'model/conv_lst_m2d/split/ReadVariableOp?)model/conv_lst_m2d/split_1/ReadVariableOp?)model/conv_lst_m2d/split_2/ReadVariableOp?model/conv_lst_m2d/while?)model/conv_lst_m2d_1/split/ReadVariableOp?+model/conv_lst_m2d_1/split_1/ReadVariableOp?+model/conv_lst_m2d_1/split_2/ReadVariableOp?model/conv_lst_m2d_1/while?
model/conv_lst_m2d/zeros_like	ZerosLikeinput_1*
T0*>
_output_shapes,
*:(????????????????????2
model/conv_lst_m2d/zeros_like?
(model/conv_lst_m2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/conv_lst_m2d/Sum/reduction_indices?
model/conv_lst_m2d/SumSum!model/conv_lst_m2d/zeros_like:y:01model/conv_lst_m2d/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
model/conv_lst_m2d/Sum?
model/conv_lst_m2d/zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
model/conv_lst_m2d/zeros?
model/conv_lst_m2d/convolutionConv2Dmodel/conv_lst_m2d/Sum:output:0!model/conv_lst_m2d/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2 
model/conv_lst_m2d/convolution?
!model/conv_lst_m2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2#
!model/conv_lst_m2d/transpose/perm?
model/conv_lst_m2d/transpose	Transposeinput_1*model/conv_lst_m2d/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
model/conv_lst_m2d/transpose?
model/conv_lst_m2d/ShapeShape model/conv_lst_m2d/transpose:y:0*
T0*
_output_shapes
:2
model/conv_lst_m2d/Shape?
&model/conv_lst_m2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model/conv_lst_m2d/strided_slice/stack?
(model/conv_lst_m2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/conv_lst_m2d/strided_slice/stack_1?
(model/conv_lst_m2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/conv_lst_m2d/strided_slice/stack_2?
 model/conv_lst_m2d/strided_sliceStridedSlice!model/conv_lst_m2d/Shape:output:0/model/conv_lst_m2d/strided_slice/stack:output:01model/conv_lst_m2d/strided_slice/stack_1:output:01model/conv_lst_m2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model/conv_lst_m2d/strided_slice?
.model/conv_lst_m2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.model/conv_lst_m2d/TensorArrayV2/element_shape?
 model/conv_lst_m2d/TensorArrayV2TensorListReserve7model/conv_lst_m2d/TensorArrayV2/element_shape:output:0)model/conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 model/conv_lst_m2d/TensorArrayV2?
Hmodel/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2J
Hmodel/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape?
:model/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor model/conv_lst_m2d/transpose:y:0Qmodel/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:model/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor?
(model/conv_lst_m2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/conv_lst_m2d/strided_slice_1/stack?
*model/conv_lst_m2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/conv_lst_m2d/strided_slice_1/stack_1?
*model/conv_lst_m2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/conv_lst_m2d/strided_slice_1/stack_2?
"model/conv_lst_m2d/strided_slice_1StridedSlice model/conv_lst_m2d/transpose:y:01model/conv_lst_m2d/strided_slice_1/stack:output:03model/conv_lst_m2d/strided_slice_1/stack_1:output:03model/conv_lst_m2d/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2$
"model/conv_lst_m2d/strided_slice_1?
"model/conv_lst_m2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/conv_lst_m2d/split/split_dim?
'model/conv_lst_m2d/split/ReadVariableOpReadVariableOp0model_conv_lst_m2d_split_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'model/conv_lst_m2d/split/ReadVariableOp?
model/conv_lst_m2d/splitSplit+model/conv_lst_m2d/split/split_dim:output:0/model/conv_lst_m2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
model/conv_lst_m2d/split?
$model/conv_lst_m2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv_lst_m2d/split_1/split_dim?
)model/conv_lst_m2d/split_1/ReadVariableOpReadVariableOp2model_conv_lst_m2d_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02+
)model/conv_lst_m2d/split_1/ReadVariableOp?
model/conv_lst_m2d/split_1Split-model/conv_lst_m2d/split_1/split_dim:output:01model/conv_lst_m2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
model/conv_lst_m2d/split_1?
$model/conv_lst_m2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv_lst_m2d/split_2/split_dim?
)model/conv_lst_m2d/split_2/ReadVariableOpReadVariableOp2model_conv_lst_m2d_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/conv_lst_m2d/split_2/ReadVariableOp?
model/conv_lst_m2d/split_2Split-model/conv_lst_m2d/split_2/split_dim:output:01model/conv_lst_m2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
model/conv_lst_m2d/split_2?
 model/conv_lst_m2d/convolution_1Conv2D+model/conv_lst_m2d/strided_slice_1:output:0!model/conv_lst_m2d/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_1?
model/conv_lst_m2d/BiasAddBiasAdd)model/conv_lst_m2d/convolution_1:output:0#model/conv_lst_m2d/split_2:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/BiasAdd?
 model/conv_lst_m2d/convolution_2Conv2D+model/conv_lst_m2d/strided_slice_1:output:0!model/conv_lst_m2d/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_2?
model/conv_lst_m2d/BiasAdd_1BiasAdd)model/conv_lst_m2d/convolution_2:output:0#model/conv_lst_m2d/split_2:output:1*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/BiasAdd_1?
 model/conv_lst_m2d/convolution_3Conv2D+model/conv_lst_m2d/strided_slice_1:output:0!model/conv_lst_m2d/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_3?
model/conv_lst_m2d/BiasAdd_2BiasAdd)model/conv_lst_m2d/convolution_3:output:0#model/conv_lst_m2d/split_2:output:2*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/BiasAdd_2?
 model/conv_lst_m2d/convolution_4Conv2D+model/conv_lst_m2d/strided_slice_1:output:0!model/conv_lst_m2d/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_4?
model/conv_lst_m2d/BiasAdd_3BiasAdd)model/conv_lst_m2d/convolution_4:output:0#model/conv_lst_m2d/split_2:output:3*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/BiasAdd_3?
 model/conv_lst_m2d/convolution_5Conv2D'model/conv_lst_m2d/convolution:output:0#model/conv_lst_m2d/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_5?
 model/conv_lst_m2d/convolution_6Conv2D'model/conv_lst_m2d/convolution:output:0#model/conv_lst_m2d/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_6?
 model/conv_lst_m2d/convolution_7Conv2D'model/conv_lst_m2d/convolution:output:0#model/conv_lst_m2d/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_7?
 model/conv_lst_m2d/convolution_8Conv2D'model/conv_lst_m2d/convolution:output:0#model/conv_lst_m2d/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d/convolution_8?
model/conv_lst_m2d/addAddV2#model/conv_lst_m2d/BiasAdd:output:0)model/conv_lst_m2d/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/addy
model/conv_lst_m2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d/Const}
model/conv_lst_m2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d/Const_1?
model/conv_lst_m2d/MulMulmodel/conv_lst_m2d/add:z:0!model/conv_lst_m2d/Const:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Mul?
model/conv_lst_m2d/Add_1AddV2model/conv_lst_m2d/Mul:z:0#model/conv_lst_m2d/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Add_1?
*model/conv_lst_m2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*model/conv_lst_m2d/clip_by_value/Minimum/y?
(model/conv_lst_m2d/clip_by_value/MinimumMinimummodel/conv_lst_m2d/Add_1:z:03model/conv_lst_m2d/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2*
(model/conv_lst_m2d/clip_by_value/Minimum?
"model/conv_lst_m2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model/conv_lst_m2d/clip_by_value/y?
 model/conv_lst_m2d/clip_by_valueMaximum,model/conv_lst_m2d/clip_by_value/Minimum:z:0+model/conv_lst_m2d/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d/clip_by_value?
model/conv_lst_m2d/add_2AddV2%model/conv_lst_m2d/BiasAdd_1:output:0)model/conv_lst_m2d/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/add_2}
model/conv_lst_m2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d/Const_2}
model/conv_lst_m2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d/Const_3?
model/conv_lst_m2d/Mul_1Mulmodel/conv_lst_m2d/add_2:z:0#model/conv_lst_m2d/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Mul_1?
model/conv_lst_m2d/Add_3AddV2model/conv_lst_m2d/Mul_1:z:0#model/conv_lst_m2d/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Add_3?
,model/conv_lst_m2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,model/conv_lst_m2d/clip_by_value_1/Minimum/y?
*model/conv_lst_m2d/clip_by_value_1/MinimumMinimummodel/conv_lst_m2d/Add_3:z:05model/conv_lst_m2d/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*model/conv_lst_m2d/clip_by_value_1/Minimum?
$model/conv_lst_m2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$model/conv_lst_m2d/clip_by_value_1/y?
"model/conv_lst_m2d/clip_by_value_1Maximum.model/conv_lst_m2d/clip_by_value_1/Minimum:z:0-model/conv_lst_m2d/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d/clip_by_value_1?
model/conv_lst_m2d/mul_2Mul&model/conv_lst_m2d/clip_by_value_1:z:0'model/conv_lst_m2d/convolution:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/mul_2?
model/conv_lst_m2d/add_4AddV2%model/conv_lst_m2d/BiasAdd_2:output:0)model/conv_lst_m2d/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/add_4?
model/conv_lst_m2d/ReluRelumodel/conv_lst_m2d/add_4:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Relu?
model/conv_lst_m2d/mul_3Mul$model/conv_lst_m2d/clip_by_value:z:0%model/conv_lst_m2d/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/mul_3?
model/conv_lst_m2d/add_5AddV2model/conv_lst_m2d/mul_2:z:0model/conv_lst_m2d/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/add_5?
model/conv_lst_m2d/add_6AddV2%model/conv_lst_m2d/BiasAdd_3:output:0)model/conv_lst_m2d/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/add_6}
model/conv_lst_m2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d/Const_4}
model/conv_lst_m2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d/Const_5?
model/conv_lst_m2d/Mul_4Mulmodel/conv_lst_m2d/add_6:z:0#model/conv_lst_m2d/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Mul_4?
model/conv_lst_m2d/Add_7AddV2model/conv_lst_m2d/Mul_4:z:0#model/conv_lst_m2d/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Add_7?
,model/conv_lst_m2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,model/conv_lst_m2d/clip_by_value_2/Minimum/y?
*model/conv_lst_m2d/clip_by_value_2/MinimumMinimummodel/conv_lst_m2d/Add_7:z:05model/conv_lst_m2d/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*model/conv_lst_m2d/clip_by_value_2/Minimum?
$model/conv_lst_m2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$model/conv_lst_m2d/clip_by_value_2/y?
"model/conv_lst_m2d/clip_by_value_2Maximum.model/conv_lst_m2d/clip_by_value_2/Minimum:z:0-model/conv_lst_m2d/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d/clip_by_value_2?
model/conv_lst_m2d/Relu_1Relumodel/conv_lst_m2d/add_5:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/Relu_1?
model/conv_lst_m2d/mul_5Mul&model/conv_lst_m2d/clip_by_value_2:z:0'model/conv_lst_m2d/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/mul_5?
0model/conv_lst_m2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0model/conv_lst_m2d/TensorArrayV2_1/element_shape?
"model/conv_lst_m2d/TensorArrayV2_1TensorListReserve9model/conv_lst_m2d/TensorArrayV2_1/element_shape:output:0)model/conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model/conv_lst_m2d/TensorArrayV2_1t
model/conv_lst_m2d/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/conv_lst_m2d/time?
+model/conv_lst_m2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+model/conv_lst_m2d/while/maximum_iterations?
%model/conv_lst_m2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/conv_lst_m2d/while/loop_counter?
model/conv_lst_m2d/whileWhile.model/conv_lst_m2d/while/loop_counter:output:04model/conv_lst_m2d/while/maximum_iterations:output:0 model/conv_lst_m2d/time:output:0+model/conv_lst_m2d/TensorArrayV2_1:handle:0'model/conv_lst_m2d/convolution:output:0'model/conv_lst_m2d/convolution:output:0)model/conv_lst_m2d/strided_slice:output:0Jmodel/conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor:output_handle:00model_conv_lst_m2d_split_readvariableop_resource2model_conv_lst_m2d_split_1_readvariableop_resource2model_conv_lst_m2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#model_conv_lst_m2d_while_body_24930*/
cond'R%
#model_conv_lst_m2d_while_cond_24929*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
model/conv_lst_m2d/while?
Cmodel/conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2E
Cmodel/conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape?
5model/conv_lst_m2d/TensorArrayV2Stack/TensorListStackTensorListStack!model/conv_lst_m2d/while:output:3Lmodel/conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype027
5model/conv_lst_m2d/TensorArrayV2Stack/TensorListStack?
(model/conv_lst_m2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(model/conv_lst_m2d/strided_slice_2/stack?
*model/conv_lst_m2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_lst_m2d/strided_slice_2/stack_1?
*model/conv_lst_m2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/conv_lst_m2d/strided_slice_2/stack_2?
"model/conv_lst_m2d/strided_slice_2StridedSlice>model/conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:01model/conv_lst_m2d/strided_slice_2/stack:output:03model/conv_lst_m2d/strided_slice_2/stack_1:output:03model/conv_lst_m2d/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2$
"model/conv_lst_m2d/strided_slice_2?
#model/conv_lst_m2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2%
#model/conv_lst_m2d/transpose_1/perm?
model/conv_lst_m2d/transpose_1	Transpose>model/conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:0,model/conv_lst_m2d/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2 
model/conv_lst_m2d/transpose_1?
model/conv_lst_m2d_1/zeros_like	ZerosLike"model/conv_lst_m2d/transpose_1:y:0*
T0*>
_output_shapes,
*:(????????????????????@2!
model/conv_lst_m2d_1/zeros_like?
*model/conv_lst_m2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*model/conv_lst_m2d_1/Sum/reduction_indices?
model/conv_lst_m2d_1/SumSum#model/conv_lst_m2d_1/zeros_like:y:03model/conv_lst_m2d_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Sum?
*model/conv_lst_m2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2,
*model/conv_lst_m2d_1/zeros/shape_as_tensor?
 model/conv_lst_m2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model/conv_lst_m2d_1/zeros/Const?
model/conv_lst_m2d_1/zerosFill3model/conv_lst_m2d_1/zeros/shape_as_tensor:output:0)model/conv_lst_m2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:@@2
model/conv_lst_m2d_1/zeros?
 model/conv_lst_m2d_1/convolutionConv2D!model/conv_lst_m2d_1/Sum:output:0#model/conv_lst_m2d_1/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 model/conv_lst_m2d_1/convolution?
#model/conv_lst_m2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2%
#model/conv_lst_m2d_1/transpose/perm?
model/conv_lst_m2d_1/transpose	Transpose"model/conv_lst_m2d/transpose_1:y:0,model/conv_lst_m2d_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2 
model/conv_lst_m2d_1/transpose?
model/conv_lst_m2d_1/ShapeShape"model/conv_lst_m2d_1/transpose:y:0*
T0*
_output_shapes
:2
model/conv_lst_m2d_1/Shape?
(model/conv_lst_m2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/conv_lst_m2d_1/strided_slice/stack?
*model/conv_lst_m2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/conv_lst_m2d_1/strided_slice/stack_1?
*model/conv_lst_m2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/conv_lst_m2d_1/strided_slice/stack_2?
"model/conv_lst_m2d_1/strided_sliceStridedSlice#model/conv_lst_m2d_1/Shape:output:01model/conv_lst_m2d_1/strided_slice/stack:output:03model/conv_lst_m2d_1/strided_slice/stack_1:output:03model/conv_lst_m2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model/conv_lst_m2d_1/strided_slice?
0model/conv_lst_m2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model/conv_lst_m2d_1/TensorArrayV2/element_shape?
"model/conv_lst_m2d_1/TensorArrayV2TensorListReserve9model/conv_lst_m2d_1/TensorArrayV2/element_shape:output:0+model/conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model/conv_lst_m2d_1/TensorArrayV2?
Jmodel/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2L
Jmodel/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
<model/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model/conv_lst_m2d_1/transpose:y:0Smodel/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<model/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor?
*model/conv_lst_m2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_lst_m2d_1/strided_slice_1/stack?
,model/conv_lst_m2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_lst_m2d_1/strided_slice_1/stack_1?
,model/conv_lst_m2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_lst_m2d_1/strided_slice_1/stack_2?
$model/conv_lst_m2d_1/strided_slice_1StridedSlice"model/conv_lst_m2d_1/transpose:y:03model/conv_lst_m2d_1/strided_slice_1/stack:output:05model/conv_lst_m2d_1/strided_slice_1/stack_1:output:05model/conv_lst_m2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2&
$model/conv_lst_m2d_1/strided_slice_1?
$model/conv_lst_m2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv_lst_m2d_1/split/split_dim?
)model/conv_lst_m2d_1/split/ReadVariableOpReadVariableOp2model_conv_lst_m2d_1_split_readvariableop_resource*'
_output_shapes
:@?*
dtype02+
)model/conv_lst_m2d_1/split/ReadVariableOp?
model/conv_lst_m2d_1/splitSplit-model/conv_lst_m2d_1/split/split_dim:output:01model/conv_lst_m2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
model/conv_lst_m2d_1/split?
&model/conv_lst_m2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model/conv_lst_m2d_1/split_1/split_dim?
+model/conv_lst_m2d_1/split_1/ReadVariableOpReadVariableOp4model_conv_lst_m2d_1_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+model/conv_lst_m2d_1/split_1/ReadVariableOp?
model/conv_lst_m2d_1/split_1Split/model/conv_lst_m2d_1/split_1/split_dim:output:03model/conv_lst_m2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
model/conv_lst_m2d_1/split_1?
&model/conv_lst_m2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv_lst_m2d_1/split_2/split_dim?
+model/conv_lst_m2d_1/split_2/ReadVariableOpReadVariableOp4model_conv_lst_m2d_1_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+model/conv_lst_m2d_1/split_2/ReadVariableOp?
model/conv_lst_m2d_1/split_2Split/model/conv_lst_m2d_1/split_2/split_dim:output:03model/conv_lst_m2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
model/conv_lst_m2d_1/split_2?
"model/conv_lst_m2d_1/convolution_1Conv2D-model/conv_lst_m2d_1/strided_slice_1:output:0#model/conv_lst_m2d_1/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_1?
model/conv_lst_m2d_1/BiasAddBiasAdd+model/conv_lst_m2d_1/convolution_1:output:0%model/conv_lst_m2d_1/split_2:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/BiasAdd?
"model/conv_lst_m2d_1/convolution_2Conv2D-model/conv_lst_m2d_1/strided_slice_1:output:0#model/conv_lst_m2d_1/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_2?
model/conv_lst_m2d_1/BiasAdd_1BiasAdd+model/conv_lst_m2d_1/convolution_2:output:0%model/conv_lst_m2d_1/split_2:output:1*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d_1/BiasAdd_1?
"model/conv_lst_m2d_1/convolution_3Conv2D-model/conv_lst_m2d_1/strided_slice_1:output:0#model/conv_lst_m2d_1/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_3?
model/conv_lst_m2d_1/BiasAdd_2BiasAdd+model/conv_lst_m2d_1/convolution_3:output:0%model/conv_lst_m2d_1/split_2:output:2*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d_1/BiasAdd_2?
"model/conv_lst_m2d_1/convolution_4Conv2D-model/conv_lst_m2d_1/strided_slice_1:output:0#model/conv_lst_m2d_1/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_4?
model/conv_lst_m2d_1/BiasAdd_3BiasAdd+model/conv_lst_m2d_1/convolution_4:output:0%model/conv_lst_m2d_1/split_2:output:3*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d_1/BiasAdd_3?
"model/conv_lst_m2d_1/convolution_5Conv2D)model/conv_lst_m2d_1/convolution:output:0%model/conv_lst_m2d_1/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_5?
"model/conv_lst_m2d_1/convolution_6Conv2D)model/conv_lst_m2d_1/convolution:output:0%model/conv_lst_m2d_1/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_6?
"model/conv_lst_m2d_1/convolution_7Conv2D)model/conv_lst_m2d_1/convolution:output:0%model/conv_lst_m2d_1/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_7?
"model/conv_lst_m2d_1/convolution_8Conv2D)model/conv_lst_m2d_1/convolution:output:0%model/conv_lst_m2d_1/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model/conv_lst_m2d_1/convolution_8?
model/conv_lst_m2d_1/addAddV2%model/conv_lst_m2d_1/BiasAdd:output:0+model/conv_lst_m2d_1/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/add}
model/conv_lst_m2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d_1/Const?
model/conv_lst_m2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d_1/Const_1?
model/conv_lst_m2d_1/MulMulmodel/conv_lst_m2d_1/add:z:0#model/conv_lst_m2d_1/Const:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Mul?
model/conv_lst_m2d_1/Add_1AddV2model/conv_lst_m2d_1/Mul:z:0%model/conv_lst_m2d_1/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Add_1?
,model/conv_lst_m2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,model/conv_lst_m2d_1/clip_by_value/Minimum/y?
*model/conv_lst_m2d_1/clip_by_value/MinimumMinimummodel/conv_lst_m2d_1/Add_1:z:05model/conv_lst_m2d_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*model/conv_lst_m2d_1/clip_by_value/Minimum?
$model/conv_lst_m2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$model/conv_lst_m2d_1/clip_by_value/y?
"model/conv_lst_m2d_1/clip_by_valueMaximum.model/conv_lst_m2d_1/clip_by_value/Minimum:z:0-model/conv_lst_m2d_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d_1/clip_by_value?
model/conv_lst_m2d_1/add_2AddV2'model/conv_lst_m2d_1/BiasAdd_1:output:0+model/conv_lst_m2d_1/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/add_2?
model/conv_lst_m2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d_1/Const_2?
model/conv_lst_m2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d_1/Const_3?
model/conv_lst_m2d_1/Mul_1Mulmodel/conv_lst_m2d_1/add_2:z:0%model/conv_lst_m2d_1/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Mul_1?
model/conv_lst_m2d_1/Add_3AddV2model/conv_lst_m2d_1/Mul_1:z:0%model/conv_lst_m2d_1/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Add_3?
.model/conv_lst_m2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.model/conv_lst_m2d_1/clip_by_value_1/Minimum/y?
,model/conv_lst_m2d_1/clip_by_value_1/MinimumMinimummodel/conv_lst_m2d_1/Add_3:z:07model/conv_lst_m2d_1/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,model/conv_lst_m2d_1/clip_by_value_1/Minimum?
&model/conv_lst_m2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&model/conv_lst_m2d_1/clip_by_value_1/y?
$model/conv_lst_m2d_1/clip_by_value_1Maximum0model/conv_lst_m2d_1/clip_by_value_1/Minimum:z:0/model/conv_lst_m2d_1/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2&
$model/conv_lst_m2d_1/clip_by_value_1?
model/conv_lst_m2d_1/mul_2Mul(model/conv_lst_m2d_1/clip_by_value_1:z:0)model/conv_lst_m2d_1/convolution:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/mul_2?
model/conv_lst_m2d_1/add_4AddV2'model/conv_lst_m2d_1/BiasAdd_2:output:0+model/conv_lst_m2d_1/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/add_4?
model/conv_lst_m2d_1/ReluRelumodel/conv_lst_m2d_1/add_4:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Relu?
model/conv_lst_m2d_1/mul_3Mul&model/conv_lst_m2d_1/clip_by_value:z:0'model/conv_lst_m2d_1/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/mul_3?
model/conv_lst_m2d_1/add_5AddV2model/conv_lst_m2d_1/mul_2:z:0model/conv_lst_m2d_1/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/add_5?
model/conv_lst_m2d_1/add_6AddV2'model/conv_lst_m2d_1/BiasAdd_3:output:0+model/conv_lst_m2d_1/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/add_6?
model/conv_lst_m2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/conv_lst_m2d_1/Const_4?
model/conv_lst_m2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/conv_lst_m2d_1/Const_5?
model/conv_lst_m2d_1/Mul_4Mulmodel/conv_lst_m2d_1/add_6:z:0%model/conv_lst_m2d_1/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Mul_4?
model/conv_lst_m2d_1/Add_7AddV2model/conv_lst_m2d_1/Mul_4:z:0%model/conv_lst_m2d_1/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Add_7?
.model/conv_lst_m2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.model/conv_lst_m2d_1/clip_by_value_2/Minimum/y?
,model/conv_lst_m2d_1/clip_by_value_2/MinimumMinimummodel/conv_lst_m2d_1/Add_7:z:07model/conv_lst_m2d_1/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2.
,model/conv_lst_m2d_1/clip_by_value_2/Minimum?
&model/conv_lst_m2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&model/conv_lst_m2d_1/clip_by_value_2/y?
$model/conv_lst_m2d_1/clip_by_value_2Maximum0model/conv_lst_m2d_1/clip_by_value_2/Minimum:z:0/model/conv_lst_m2d_1/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2&
$model/conv_lst_m2d_1/clip_by_value_2?
model/conv_lst_m2d_1/Relu_1Relumodel/conv_lst_m2d_1/add_5:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/Relu_1?
model/conv_lst_m2d_1/mul_5Mul(model/conv_lst_m2d_1/clip_by_value_2:z:0)model/conv_lst_m2d_1/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d_1/mul_5?
2model/conv_lst_m2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   24
2model/conv_lst_m2d_1/TensorArrayV2_1/element_shape?
$model/conv_lst_m2d_1/TensorArrayV2_1TensorListReserve;model/conv_lst_m2d_1/TensorArrayV2_1/element_shape:output:0+model/conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model/conv_lst_m2d_1/TensorArrayV2_1x
model/conv_lst_m2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/conv_lst_m2d_1/time?
-model/conv_lst_m2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model/conv_lst_m2d_1/while/maximum_iterations?
'model/conv_lst_m2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model/conv_lst_m2d_1/while/loop_counter?
model/conv_lst_m2d_1/whileWhile0model/conv_lst_m2d_1/while/loop_counter:output:06model/conv_lst_m2d_1/while/maximum_iterations:output:0"model/conv_lst_m2d_1/time:output:0-model/conv_lst_m2d_1/TensorArrayV2_1:handle:0)model/conv_lst_m2d_1/convolution:output:0)model/conv_lst_m2d_1/convolution:output:0+model/conv_lst_m2d_1/strided_slice:output:0Lmodel/conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:02model_conv_lst_m2d_1_split_readvariableop_resource4model_conv_lst_m2d_1_split_1_readvariableop_resource4model_conv_lst_m2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%model_conv_lst_m2d_1_while_body_25148*1
cond)R'
%model_conv_lst_m2d_1_while_cond_25147*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
model/conv_lst_m2d_1/while?
Emodel/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2G
Emodel/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape?
7model/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStackTensorListStack#model/conv_lst_m2d_1/while:output:3Nmodel/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype029
7model/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack?
*model/conv_lst_m2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model/conv_lst_m2d_1/strided_slice_2/stack?
,model/conv_lst_m2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_lst_m2d_1/strided_slice_2/stack_1?
,model/conv_lst_m2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_lst_m2d_1/strided_slice_2/stack_2?
$model/conv_lst_m2d_1/strided_slice_2StridedSlice@model/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:03model/conv_lst_m2d_1/strided_slice_2/stack:output:05model/conv_lst_m2d_1/strided_slice_2/stack_1:output:05model/conv_lst_m2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2&
$model/conv_lst_m2d_1/strided_slice_2?
%model/conv_lst_m2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2'
%model/conv_lst_m2d_1/transpose_1/perm?
 model/conv_lst_m2d_1/transpose_1	Transpose@model/conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:0.model/conv_lst_m2d_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2"
 model/conv_lst_m2d_1/transpose_1?
IdentityIdentity$model/conv_lst_m2d_1/transpose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp(^model/conv_lst_m2d/split/ReadVariableOp*^model/conv_lst_m2d/split_1/ReadVariableOp*^model/conv_lst_m2d/split_2/ReadVariableOp^model/conv_lst_m2d/while*^model/conv_lst_m2d_1/split/ReadVariableOp,^model/conv_lst_m2d_1/split_1/ReadVariableOp,^model/conv_lst_m2d_1/split_2/ReadVariableOp^model/conv_lst_m2d_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2R
'model/conv_lst_m2d/split/ReadVariableOp'model/conv_lst_m2d/split/ReadVariableOp2V
)model/conv_lst_m2d/split_1/ReadVariableOp)model/conv_lst_m2d/split_1/ReadVariableOp2V
)model/conv_lst_m2d/split_2/ReadVariableOp)model/conv_lst_m2d/split_2/ReadVariableOp24
model/conv_lst_m2d/whilemodel/conv_lst_m2d/while2V
)model/conv_lst_m2d_1/split/ReadVariableOp)model/conv_lst_m2d_1/split/ReadVariableOp2Z
+model/conv_lst_m2d_1/split_1/ReadVariableOp+model/conv_lst_m2d_1/split_1/ReadVariableOp2Z
+model/conv_lst_m2d_1/split_2/ReadVariableOp+model/conv_lst_m2d_1/split_2/ReadVariableOp28
model/conv_lst_m2d_1/whilemodel/conv_lst_m2d_1/while:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1
?f
?
while_body_28721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_conv_lst_m2d_layer_call_fn_28594
inputs_0"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_254582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
>
_output_shapes,
*:(????????????????????
"
_user_specified_name
inputs/0
?
?
#model_conv_lst_m2d_while_cond_24929B
>model_conv_lst_m2d_while_model_conv_lst_m2d_while_loop_counterH
Dmodel_conv_lst_m2d_while_model_conv_lst_m2d_while_maximum_iterations(
$model_conv_lst_m2d_while_placeholder*
&model_conv_lst_m2d_while_placeholder_1*
&model_conv_lst_m2d_while_placeholder_2*
&model_conv_lst_m2d_while_placeholder_3B
>model_conv_lst_m2d_while_less_model_conv_lst_m2d_strided_sliceY
Umodel_conv_lst_m2d_while_model_conv_lst_m2d_while_cond_24929___redundant_placeholder0Y
Umodel_conv_lst_m2d_while_model_conv_lst_m2d_while_cond_24929___redundant_placeholder1Y
Umodel_conv_lst_m2d_while_model_conv_lst_m2d_while_cond_24929___redundant_placeholder2Y
Umodel_conv_lst_m2d_while_model_conv_lst_m2d_while_cond_24929___redundant_placeholder3%
!model_conv_lst_m2d_while_identity
?
model/conv_lst_m2d/while/LessLess$model_conv_lst_m2d_while_placeholder>model_conv_lst_m2d_while_less_model_conv_lst_m2d_strided_slice*
T0*
_output_shapes
: 2
model/conv_lst_m2d/while/Less?
!model/conv_lst_m2d/while/IdentityIdentity!model/conv_lst_m2d/while/Less:z:0*
T0
*
_output_shapes
: 2#
!model/conv_lst_m2d/while/Identity"O
!model_conv_lst_m2d_while_identity*model/conv_lst_m2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_30312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_30312___redundant_placeholder03
/while_while_cond_30312___redundant_placeholder13
/while_while_cond_30312___redundant_placeholder23
/while_while_cond_30312___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_26677
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_26677___redundant_placeholder03
/while_while_cond_26677___redundant_placeholder13
/while_while_cond_26677___redundant_placeholder23
/while_while_cond_26677___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
@__inference_model_layer_call_and_return_conditional_losses_27042

inputs-
conv_lst_m2d_26805:?-
conv_lst_m2d_26807:@?!
conv_lst_m2d_26809:	?/
conv_lst_m2d_1_27034:@?/
conv_lst_m2d_1_27036:@?#
conv_lst_m2d_1_27038:	?
identity??$conv_lst_m2d/StatefulPartitionedCall?&conv_lst_m2d_1/StatefulPartitionedCall?
$conv_lst_m2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lst_m2d_26805conv_lst_m2d_26807conv_lst_m2d_26809*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_268042&
$conv_lst_m2d/StatefulPartitionedCall?
&conv_lst_m2d_1/StatefulPartitionedCallStatefulPartitionedCall-conv_lst_m2d/StatefulPartitionedCall:output:0conv_lst_m2d_1_27034conv_lst_m2d_1_27036conv_lst_m2d_1_27038*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_270332(
&conv_lst_m2d_1/StatefulPartitionedCall?
IdentityIdentity/conv_lst_m2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp%^conv_lst_m2d/StatefulPartitionedCall'^conv_lst_m2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2L
$conv_lst_m2d/StatefulPartitionedCall$conv_lst_m2d/StatefulPartitionedCall2P
&conv_lst_m2d_1/StatefulPartitionedCall&conv_lst_m2d_1/StatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30807

inputs
states_0
states_18
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?=
?
__inference__traced_save_30905
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_conv_lst_m2d_kernel_read_readvariableop<
8savev2_conv_lst_m2d_recurrent_kernel_read_readvariableop0
,savev2_conv_lst_m2d_bias_read_readvariableop4
0savev2_conv_lst_m2d_1_kernel_read_readvariableop>
:savev2_conv_lst_m2d_1_recurrent_kernel_read_readvariableop2
.savev2_conv_lst_m2d_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_conv_lst_m2d_kernel_m_read_readvariableopC
?savev2_adam_conv_lst_m2d_recurrent_kernel_m_read_readvariableop7
3savev2_adam_conv_lst_m2d_bias_m_read_readvariableop;
7savev2_adam_conv_lst_m2d_1_kernel_m_read_readvariableopE
Asavev2_adam_conv_lst_m2d_1_recurrent_kernel_m_read_readvariableop9
5savev2_adam_conv_lst_m2d_1_bias_m_read_readvariableop9
5savev2_adam_conv_lst_m2d_kernel_v_read_readvariableopC
?savev2_adam_conv_lst_m2d_recurrent_kernel_v_read_readvariableop7
3savev2_adam_conv_lst_m2d_bias_v_read_readvariableop;
7savev2_adam_conv_lst_m2d_1_kernel_v_read_readvariableopE
Asavev2_adam_conv_lst_m2d_1_recurrent_kernel_v_read_readvariableop9
5savev2_adam_conv_lst_m2d_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_conv_lst_m2d_kernel_read_readvariableop8savev2_conv_lst_m2d_recurrent_kernel_read_readvariableop,savev2_conv_lst_m2d_bias_read_readvariableop0savev2_conv_lst_m2d_1_kernel_read_readvariableop:savev2_conv_lst_m2d_1_recurrent_kernel_read_readvariableop.savev2_conv_lst_m2d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_conv_lst_m2d_kernel_m_read_readvariableop?savev2_adam_conv_lst_m2d_recurrent_kernel_m_read_readvariableop3savev2_adam_conv_lst_m2d_bias_m_read_readvariableop7savev2_adam_conv_lst_m2d_1_kernel_m_read_readvariableopAsavev2_adam_conv_lst_m2d_1_recurrent_kernel_m_read_readvariableop5savev2_adam_conv_lst_m2d_1_bias_m_read_readvariableop5savev2_adam_conv_lst_m2d_kernel_v_read_readvariableop?savev2_adam_conv_lst_m2d_recurrent_kernel_v_read_readvariableop3savev2_adam_conv_lst_m2d_bias_v_read_readvariableop7savev2_adam_conv_lst_m2d_1_kernel_v_read_readvariableopAsavev2_adam_conv_lst_m2d_1_recurrent_kernel_v_read_readvariableop5savev2_adam_conv_lst_m2d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :?:@?:?:@?:@?:?: : :?:@?:?:@?:@?:?:?:@?:?:@?:@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:-	)
'
_output_shapes
:@?:-
)
'
_output_shapes
:@?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:-)
'
_output_shapes
:?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:

_output_shapes
: 
?
?
while_cond_26906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_26906___redundant_placeholder03
/while_while_cond_26906___redundant_placeholder13
/while_while_cond_26906___redundant_placeholder23
/while_while_cond_26906___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_27033

inputs8
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26907*
condR
while_cond_26906*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?	
?
.__inference_conv_lst_m2d_1_layer_call_fn_29529
inputs_0"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_263462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
>
_output_shapes,
*:(????????????????????@
"
_user_specified_name
inputs/0
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29287

inputs8
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_29161*
condR
while_cond_29160*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?$
?
while_body_25626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_25650_0:?(
while_25652_0:@?
while_25654_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_25650:?&
while_25652:@?
while_25654:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_25650_0while_25652_0while_25654_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_255642
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5z

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"
while_25650while_25650_0"
while_25652while_25652_0"
while_25654while_25654_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_28847
inputs_08
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilex

zeros_like	ZerosLikeinputs_0*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_28721*
condR
while_cond_28720*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:h d
>
_output_shapes,
*:(????????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_29160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_29160___redundant_placeholder03
/while_while_cond_29160___redundant_placeholder13
/while_while_cond_29160___redundant_placeholder23
/while_while_cond_29160___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_29380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_29380___redundant_placeholder03
/while_while_cond_29380___redundant_placeholder13
/while_while_cond_29380___redundant_placeholder23
/while_while_cond_29380___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
??
?
@__inference_model_layer_call_and_return_conditional_losses_28583

inputsE
*conv_lst_m2d_split_readvariableop_resource:?G
,conv_lst_m2d_split_1_readvariableop_resource:@?;
,conv_lst_m2d_split_2_readvariableop_resource:	?G
,conv_lst_m2d_1_split_readvariableop_resource:@?I
.conv_lst_m2d_1_split_1_readvariableop_resource:@?=
.conv_lst_m2d_1_split_2_readvariableop_resource:	?
identity??!conv_lst_m2d/split/ReadVariableOp?#conv_lst_m2d/split_1/ReadVariableOp?#conv_lst_m2d/split_2/ReadVariableOp?conv_lst_m2d/while?#conv_lst_m2d_1/split/ReadVariableOp?%conv_lst_m2d_1/split_1/ReadVariableOp?%conv_lst_m2d_1/split_2/ReadVariableOp?conv_lst_m2d_1/while?
conv_lst_m2d/zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2
conv_lst_m2d/zeros_like?
"conv_lst_m2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"conv_lst_m2d/Sum/reduction_indices?
conv_lst_m2d/SumSumconv_lst_m2d/zeros_like:y:0+conv_lst_m2d/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
conv_lst_m2d/Sum?
conv_lst_m2d/zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
conv_lst_m2d/zeros?
conv_lst_m2d/convolutionConv2Dconv_lst_m2d/Sum:output:0conv_lst_m2d/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution?
conv_lst_m2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d/transpose/perm?
conv_lst_m2d/transpose	Transposeinputs$conv_lst_m2d/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
conv_lst_m2d/transposer
conv_lst_m2d/ShapeShapeconv_lst_m2d/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d/Shape?
 conv_lst_m2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv_lst_m2d/strided_slice/stack?
"conv_lst_m2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"conv_lst_m2d/strided_slice/stack_1?
"conv_lst_m2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"conv_lst_m2d/strided_slice/stack_2?
conv_lst_m2d/strided_sliceStridedSliceconv_lst_m2d/Shape:output:0)conv_lst_m2d/strided_slice/stack:output:0+conv_lst_m2d/strided_slice/stack_1:output:0+conv_lst_m2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d/strided_slice?
(conv_lst_m2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(conv_lst_m2d/TensorArrayV2/element_shape?
conv_lst_m2d/TensorArrayV2TensorListReserve1conv_lst_m2d/TensorArrayV2/element_shape:output:0#conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d/TensorArrayV2?
Bconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2D
Bconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape?
4conv_lst_m2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d/transpose:y:0Kconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4conv_lst_m2d/TensorArrayUnstack/TensorListFromTensor?
"conv_lst_m2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d/strided_slice_1/stack?
$conv_lst_m2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_1/stack_1?
$conv_lst_m2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_1/stack_2?
conv_lst_m2d/strided_slice_1StridedSliceconv_lst_m2d/transpose:y:0+conv_lst_m2d/strided_slice_1/stack:output:0-conv_lst_m2d/strided_slice_1/stack_1:output:0-conv_lst_m2d/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
conv_lst_m2d/strided_slice_1~
conv_lst_m2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/split/split_dim?
!conv_lst_m2d/split/ReadVariableOpReadVariableOp*conv_lst_m2d_split_readvariableop_resource*'
_output_shapes
:?*
dtype02#
!conv_lst_m2d/split/ReadVariableOp?
conv_lst_m2d/splitSplit%conv_lst_m2d/split/split_dim:output:0)conv_lst_m2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
conv_lst_m2d/split?
conv_lst_m2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d/split_1/split_dim?
#conv_lst_m2d/split_1/ReadVariableOpReadVariableOp,conv_lst_m2d_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#conv_lst_m2d/split_1/ReadVariableOp?
conv_lst_m2d/split_1Split'conv_lst_m2d/split_1/split_dim:output:0+conv_lst_m2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d/split_1?
conv_lst_m2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_lst_m2d/split_2/split_dim?
#conv_lst_m2d/split_2/ReadVariableOpReadVariableOp,conv_lst_m2d_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#conv_lst_m2d/split_2/ReadVariableOp?
conv_lst_m2d/split_2Split'conv_lst_m2d/split_2/split_dim:output:0+conv_lst_m2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d/split_2?
conv_lst_m2d/convolution_1Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_1?
conv_lst_m2d/BiasAddBiasAdd#conv_lst_m2d/convolution_1:output:0conv_lst_m2d/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd?
conv_lst_m2d/convolution_2Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_2?
conv_lst_m2d/BiasAdd_1BiasAdd#conv_lst_m2d/convolution_2:output:0conv_lst_m2d/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_1?
conv_lst_m2d/convolution_3Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_3?
conv_lst_m2d/BiasAdd_2BiasAdd#conv_lst_m2d/convolution_3:output:0conv_lst_m2d/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_2?
conv_lst_m2d/convolution_4Conv2D%conv_lst_m2d/strided_slice_1:output:0conv_lst_m2d/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_4?
conv_lst_m2d/BiasAdd_3BiasAdd#conv_lst_m2d/convolution_4:output:0conv_lst_m2d/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/BiasAdd_3?
conv_lst_m2d/convolution_5Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_5?
conv_lst_m2d/convolution_6Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_6?
conv_lst_m2d/convolution_7Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_7?
conv_lst_m2d/convolution_8Conv2D!conv_lst_m2d/convolution:output:0conv_lst_m2d/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d/convolution_8?
conv_lst_m2d/addAddV2conv_lst_m2d/BiasAdd:output:0#conv_lst_m2d/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/addm
conv_lst_m2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Constq
conv_lst_m2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_1?
conv_lst_m2d/MulMulconv_lst_m2d/add:z:0conv_lst_m2d/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul?
conv_lst_m2d/Add_1AddV2conv_lst_m2d/Mul:z:0conv_lst_m2d/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_1?
$conv_lst_m2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$conv_lst_m2d/clip_by_value/Minimum/y?
"conv_lst_m2d/clip_by_value/MinimumMinimumconv_lst_m2d/Add_1:z:0-conv_lst_m2d/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/clip_by_value/Minimum?
conv_lst_m2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d/clip_by_value/y?
conv_lst_m2d/clip_by_valueMaximum&conv_lst_m2d/clip_by_value/Minimum:z:0%conv_lst_m2d/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value?
conv_lst_m2d/add_2AddV2conv_lst_m2d/BiasAdd_1:output:0#conv_lst_m2d/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_2q
conv_lst_m2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Const_2q
conv_lst_m2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_3?
conv_lst_m2d/Mul_1Mulconv_lst_m2d/add_2:z:0conv_lst_m2d/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul_1?
conv_lst_m2d/Add_3AddV2conv_lst_m2d/Mul_1:z:0conv_lst_m2d/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_3?
&conv_lst_m2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d/clip_by_value_1/Minimum/y?
$conv_lst_m2d/clip_by_value_1/MinimumMinimumconv_lst_m2d/Add_3:z:0/conv_lst_m2d/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d/clip_by_value_1/Minimum?
conv_lst_m2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d/clip_by_value_1/y?
conv_lst_m2d/clip_by_value_1Maximum(conv_lst_m2d/clip_by_value_1/Minimum:z:0'conv_lst_m2d/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value_1?
conv_lst_m2d/mul_2Mul conv_lst_m2d/clip_by_value_1:z:0!conv_lst_m2d/convolution:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_2?
conv_lst_m2d/add_4AddV2conv_lst_m2d/BiasAdd_2:output:0#conv_lst_m2d/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_4?
conv_lst_m2d/ReluReluconv_lst_m2d/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Relu?
conv_lst_m2d/mul_3Mulconv_lst_m2d/clip_by_value:z:0conv_lst_m2d/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_3?
conv_lst_m2d/add_5AddV2conv_lst_m2d/mul_2:z:0conv_lst_m2d/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_5?
conv_lst_m2d/add_6AddV2conv_lst_m2d/BiasAdd_3:output:0#conv_lst_m2d/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/add_6q
conv_lst_m2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/Const_4q
conv_lst_m2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/Const_5?
conv_lst_m2d/Mul_4Mulconv_lst_m2d/add_6:z:0conv_lst_m2d/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Mul_4?
conv_lst_m2d/Add_7AddV2conv_lst_m2d/Mul_4:z:0conv_lst_m2d/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Add_7?
&conv_lst_m2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d/clip_by_value_2/Minimum/y?
$conv_lst_m2d/clip_by_value_2/MinimumMinimumconv_lst_m2d/Add_7:z:0/conv_lst_m2d/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d/clip_by_value_2/Minimum?
conv_lst_m2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d/clip_by_value_2/y?
conv_lst_m2d/clip_by_value_2Maximum(conv_lst_m2d/clip_by_value_2/Minimum:z:0'conv_lst_m2d/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/clip_by_value_2?
conv_lst_m2d/Relu_1Reluconv_lst_m2d/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/Relu_1?
conv_lst_m2d/mul_5Mul conv_lst_m2d/clip_by_value_2:z:0!conv_lst_m2d/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/mul_5?
*conv_lst_m2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2,
*conv_lst_m2d/TensorArrayV2_1/element_shape?
conv_lst_m2d/TensorArrayV2_1TensorListReserve3conv_lst_m2d/TensorArrayV2_1/element_shape:output:0#conv_lst_m2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d/TensorArrayV2_1h
conv_lst_m2d/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d/time?
%conv_lst_m2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv_lst_m2d/while/maximum_iterations?
conv_lst_m2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
conv_lst_m2d/while/loop_counter?
conv_lst_m2d/whileWhile(conv_lst_m2d/while/loop_counter:output:0.conv_lst_m2d/while/maximum_iterations:output:0conv_lst_m2d/time:output:0%conv_lst_m2d/TensorArrayV2_1:handle:0!conv_lst_m2d/convolution:output:0!conv_lst_m2d/convolution:output:0#conv_lst_m2d/strided_slice:output:0Dconv_lst_m2d/TensorArrayUnstack/TensorListFromTensor:output_handle:0*conv_lst_m2d_split_readvariableop_resource,conv_lst_m2d_split_1_readvariableop_resource,conv_lst_m2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
conv_lst_m2d_while_body_28239*)
cond!R
conv_lst_m2d_while_cond_28238*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
conv_lst_m2d/while?
=conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2?
=conv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape?
/conv_lst_m2d/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d/while:output:3Fconv_lst_m2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype021
/conv_lst_m2d/TensorArrayV2Stack/TensorListStack?
"conv_lst_m2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"conv_lst_m2d/strided_slice_2/stack?
$conv_lst_m2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d/strided_slice_2/stack_1?
$conv_lst_m2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d/strided_slice_2/stack_2?
conv_lst_m2d/strided_slice_2StridedSlice8conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:0+conv_lst_m2d/strided_slice_2/stack:output:0-conv_lst_m2d/strided_slice_2/stack_1:output:0-conv_lst_m2d/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
conv_lst_m2d/strided_slice_2?
conv_lst_m2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d/transpose_1/perm?
conv_lst_m2d/transpose_1	Transpose8conv_lst_m2d/TensorArrayV2Stack/TensorListStack:tensor:0&conv_lst_m2d/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d/transpose_1?
conv_lst_m2d_1/zeros_like	ZerosLikeconv_lst_m2d/transpose_1:y:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/zeros_like?
$conv_lst_m2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_1/Sum/reduction_indices?
conv_lst_m2d_1/SumSumconv_lst_m2d_1/zeros_like:y:0-conv_lst_m2d_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Sum?
$conv_lst_m2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2&
$conv_lst_m2d_1/zeros/shape_as_tensor}
conv_lst_m2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d_1/zeros/Const?
conv_lst_m2d_1/zerosFill-conv_lst_m2d_1/zeros/shape_as_tensor:output:0#conv_lst_m2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:@@2
conv_lst_m2d_1/zeros?
conv_lst_m2d_1/convolutionConv2Dconv_lst_m2d_1/Sum:output:0conv_lst_m2d_1/zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution?
conv_lst_m2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d_1/transpose/perm?
conv_lst_m2d_1/transpose	Transposeconv_lst_m2d/transpose_1:y:0&conv_lst_m2d_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/transposex
conv_lst_m2d_1/ShapeShapeconv_lst_m2d_1/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d_1/Shape?
"conv_lst_m2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d_1/strided_slice/stack?
$conv_lst_m2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_1/strided_slice/stack_1?
$conv_lst_m2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_1/strided_slice/stack_2?
conv_lst_m2d_1/strided_sliceStridedSliceconv_lst_m2d_1/Shape:output:0+conv_lst_m2d_1/strided_slice/stack:output:0-conv_lst_m2d_1/strided_slice/stack_1:output:0-conv_lst_m2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d_1/strided_slice?
*conv_lst_m2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*conv_lst_m2d_1/TensorArrayV2/element_shape?
conv_lst_m2d_1/TensorArrayV2TensorListReserve3conv_lst_m2d_1/TensorArrayV2/element_shape:output:0%conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d_1/TensorArrayV2?
Dconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2F
Dconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
6conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d_1/transpose:y:0Mconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6conv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor?
$conv_lst_m2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d_1/strided_slice_1/stack?
&conv_lst_m2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_1/stack_1?
&conv_lst_m2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_1/stack_2?
conv_lst_m2d_1/strided_slice_1StridedSliceconv_lst_m2d_1/transpose:y:0-conv_lst_m2d_1/strided_slice_1/stack:output:0/conv_lst_m2d_1/strided_slice_1/stack_1:output:0/conv_lst_m2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2 
conv_lst_m2d_1/strided_slice_1?
conv_lst_m2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d_1/split/split_dim?
#conv_lst_m2d_1/split/ReadVariableOpReadVariableOp,conv_lst_m2d_1_split_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#conv_lst_m2d_1/split/ReadVariableOp?
conv_lst_m2d_1/splitSplit'conv_lst_m2d_1/split/split_dim:output:0+conv_lst_m2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/split?
 conv_lst_m2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 conv_lst_m2d_1/split_1/split_dim?
%conv_lst_m2d_1/split_1/ReadVariableOpReadVariableOp.conv_lst_m2d_1_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%conv_lst_m2d_1/split_1/ReadVariableOp?
conv_lst_m2d_1/split_1Split)conv_lst_m2d_1/split_1/split_dim:output:0-conv_lst_m2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_1/split_1?
 conv_lst_m2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv_lst_m2d_1/split_2/split_dim?
%conv_lst_m2d_1/split_2/ReadVariableOpReadVariableOp.conv_lst_m2d_1_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%conv_lst_m2d_1/split_2/ReadVariableOp?
conv_lst_m2d_1/split_2Split)conv_lst_m2d_1/split_2/split_dim:output:0-conv_lst_m2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_1/split_2?
conv_lst_m2d_1/convolution_1Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_1?
conv_lst_m2d_1/BiasAddBiasAdd%conv_lst_m2d_1/convolution_1:output:0conv_lst_m2d_1/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd?
conv_lst_m2d_1/convolution_2Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_2?
conv_lst_m2d_1/BiasAdd_1BiasAdd%conv_lst_m2d_1/convolution_2:output:0conv_lst_m2d_1/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_1?
conv_lst_m2d_1/convolution_3Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_3?
conv_lst_m2d_1/BiasAdd_2BiasAdd%conv_lst_m2d_1/convolution_3:output:0conv_lst_m2d_1/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_2?
conv_lst_m2d_1/convolution_4Conv2D'conv_lst_m2d_1/strided_slice_1:output:0conv_lst_m2d_1/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_4?
conv_lst_m2d_1/BiasAdd_3BiasAdd%conv_lst_m2d_1/convolution_4:output:0conv_lst_m2d_1/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/BiasAdd_3?
conv_lst_m2d_1/convolution_5Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_5?
conv_lst_m2d_1/convolution_6Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_6?
conv_lst_m2d_1/convolution_7Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_7?
conv_lst_m2d_1/convolution_8Conv2D#conv_lst_m2d_1/convolution:output:0conv_lst_m2d_1/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv_lst_m2d_1/convolution_8?
conv_lst_m2d_1/addAddV2conv_lst_m2d_1/BiasAdd:output:0%conv_lst_m2d_1/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/addq
conv_lst_m2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Constu
conv_lst_m2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_1?
conv_lst_m2d_1/MulMulconv_lst_m2d_1/add:z:0conv_lst_m2d_1/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul?
conv_lst_m2d_1/Add_1AddV2conv_lst_m2d_1/Mul:z:0conv_lst_m2d_1/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_1?
&conv_lst_m2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d_1/clip_by_value/Minimum/y?
$conv_lst_m2d_1/clip_by_value/MinimumMinimumconv_lst_m2d_1/Add_1:z:0/conv_lst_m2d_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2&
$conv_lst_m2d_1/clip_by_value/Minimum?
conv_lst_m2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d_1/clip_by_value/y?
conv_lst_m2d_1/clip_by_valueMaximum(conv_lst_m2d_1/clip_by_value/Minimum:z:0'conv_lst_m2d_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/clip_by_value?
conv_lst_m2d_1/add_2AddV2!conv_lst_m2d_1/BiasAdd_1:output:0%conv_lst_m2d_1/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_2u
conv_lst_m2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Const_2u
conv_lst_m2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_3?
conv_lst_m2d_1/Mul_1Mulconv_lst_m2d_1/add_2:z:0conv_lst_m2d_1/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul_1?
conv_lst_m2d_1/Add_3AddV2conv_lst_m2d_1/Mul_1:z:0conv_lst_m2d_1/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_3?
(conv_lst_m2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_1/clip_by_value_1/Minimum/y?
&conv_lst_m2d_1/clip_by_value_1/MinimumMinimumconv_lst_m2d_1/Add_3:z:01conv_lst_m2d_1/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2(
&conv_lst_m2d_1/clip_by_value_1/Minimum?
 conv_lst_m2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_1/clip_by_value_1/y?
conv_lst_m2d_1/clip_by_value_1Maximum*conv_lst_m2d_1/clip_by_value_1/Minimum:z:0)conv_lst_m2d_1/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/clip_by_value_1?
conv_lst_m2d_1/mul_2Mul"conv_lst_m2d_1/clip_by_value_1:z:0#conv_lst_m2d_1/convolution:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_2?
conv_lst_m2d_1/add_4AddV2!conv_lst_m2d_1/BiasAdd_2:output:0%conv_lst_m2d_1/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_4?
conv_lst_m2d_1/ReluReluconv_lst_m2d_1/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Relu?
conv_lst_m2d_1/mul_3Mul conv_lst_m2d_1/clip_by_value:z:0!conv_lst_m2d_1/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_3?
conv_lst_m2d_1/add_5AddV2conv_lst_m2d_1/mul_2:z:0conv_lst_m2d_1/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_5?
conv_lst_m2d_1/add_6AddV2!conv_lst_m2d_1/BiasAdd_3:output:0%conv_lst_m2d_1/convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/add_6u
conv_lst_m2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_1/Const_4u
conv_lst_m2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_1/Const_5?
conv_lst_m2d_1/Mul_4Mulconv_lst_m2d_1/add_6:z:0conv_lst_m2d_1/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Mul_4?
conv_lst_m2d_1/Add_7AddV2conv_lst_m2d_1/Mul_4:z:0conv_lst_m2d_1/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Add_7?
(conv_lst_m2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_1/clip_by_value_2/Minimum/y?
&conv_lst_m2d_1/clip_by_value_2/MinimumMinimumconv_lst_m2d_1/Add_7:z:01conv_lst_m2d_1/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2(
&conv_lst_m2d_1/clip_by_value_2/Minimum?
 conv_lst_m2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_1/clip_by_value_2/y?
conv_lst_m2d_1/clip_by_value_2Maximum*conv_lst_m2d_1/clip_by_value_2/Minimum:z:0)conv_lst_m2d_1/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2 
conv_lst_m2d_1/clip_by_value_2?
conv_lst_m2d_1/Relu_1Reluconv_lst_m2d_1/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/Relu_1?
conv_lst_m2d_1/mul_5Mul"conv_lst_m2d_1/clip_by_value_2:z:0#conv_lst_m2d_1/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d_1/mul_5?
,conv_lst_m2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2.
,conv_lst_m2d_1/TensorArrayV2_1/element_shape?
conv_lst_m2d_1/TensorArrayV2_1TensorListReserve5conv_lst_m2d_1/TensorArrayV2_1/element_shape:output:0%conv_lst_m2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
conv_lst_m2d_1/TensorArrayV2_1l
conv_lst_m2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d_1/time?
'conv_lst_m2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'conv_lst_m2d_1/while/maximum_iterations?
!conv_lst_m2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv_lst_m2d_1/while/loop_counter?
conv_lst_m2d_1/whileWhile*conv_lst_m2d_1/while/loop_counter:output:00conv_lst_m2d_1/while/maximum_iterations:output:0conv_lst_m2d_1/time:output:0'conv_lst_m2d_1/TensorArrayV2_1:handle:0#conv_lst_m2d_1/convolution:output:0#conv_lst_m2d_1/convolution:output:0%conv_lst_m2d_1/strided_slice:output:0Fconv_lst_m2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0,conv_lst_m2d_1_split_readvariableop_resource.conv_lst_m2d_1_split_1_readvariableop_resource.conv_lst_m2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
conv_lst_m2d_1_while_body_28457*+
cond#R!
conv_lst_m2d_1_while_cond_28456*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
conv_lst_m2d_1/while?
?conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2A
?conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape?
1conv_lst_m2d_1/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d_1/while:output:3Hconv_lst_m2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype023
1conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack?
$conv_lst_m2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$conv_lst_m2d_1/strided_slice_2/stack?
&conv_lst_m2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_lst_m2d_1/strided_slice_2/stack_1?
&conv_lst_m2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_1/strided_slice_2/stack_2?
conv_lst_m2d_1/strided_slice_2StridedSlice:conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:0-conv_lst_m2d_1/strided_slice_2/stack:output:0/conv_lst_m2d_1/strided_slice_2/stack_1:output:0/conv_lst_m2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2 
conv_lst_m2d_1/strided_slice_2?
conv_lst_m2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2!
conv_lst_m2d_1/transpose_1/perm?
conv_lst_m2d_1/transpose_1	Transpose:conv_lst_m2d_1/TensorArrayV2Stack/TensorListStack:tensor:0(conv_lst_m2d_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
conv_lst_m2d_1/transpose_1?
IdentityIdentityconv_lst_m2d_1/transpose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp"^conv_lst_m2d/split/ReadVariableOp$^conv_lst_m2d/split_1/ReadVariableOp$^conv_lst_m2d/split_2/ReadVariableOp^conv_lst_m2d/while$^conv_lst_m2d_1/split/ReadVariableOp&^conv_lst_m2d_1/split_1/ReadVariableOp&^conv_lst_m2d_1/split_2/ReadVariableOp^conv_lst_m2d_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 2F
!conv_lst_m2d/split/ReadVariableOp!conv_lst_m2d/split/ReadVariableOp2J
#conv_lst_m2d/split_1/ReadVariableOp#conv_lst_m2d/split_1/ReadVariableOp2J
#conv_lst_m2d/split_2/ReadVariableOp#conv_lst_m2d/split_2/ReadVariableOp2(
conv_lst_m2d/whileconv_lst_m2d/while2J
#conv_lst_m2d_1/split/ReadVariableOp#conv_lst_m2d_1/split/ReadVariableOp2N
%conv_lst_m2d_1/split_1/ReadVariableOp%conv_lst_m2d_1/split_1/ReadVariableOp2N
%conv_lst_m2d_1/split_2/ReadVariableOp%conv_lst_m2d_1/split_2/ReadVariableOp2,
conv_lst_m2d_1/whileconv_lst_m2d_1/while:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?f
?
while_body_29869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
conv_lst_m2d_while_cond_278006
2conv_lst_m2d_while_conv_lst_m2d_while_loop_counter<
8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations"
conv_lst_m2d_while_placeholder$
 conv_lst_m2d_while_placeholder_1$
 conv_lst_m2d_while_placeholder_2$
 conv_lst_m2d_while_placeholder_36
2conv_lst_m2d_while_less_conv_lst_m2d_strided_sliceM
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_27800___redundant_placeholder0M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_27800___redundant_placeholder1M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_27800___redundant_placeholder2M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_27800___redundant_placeholder3
conv_lst_m2d_while_identity
?
conv_lst_m2d/while/LessLessconv_lst_m2d_while_placeholder2conv_lst_m2d_while_less_conv_lst_m2d_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Less?
conv_lst_m2d/while/IdentityIdentityconv_lst_m2d/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d/while/Identity"C
conv_lst_m2d_while_identity$conv_lst_m2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?	
?
#__inference_signature_wrapper_27673
input_1"
unknown:?$
	unknown_0:@?
	unknown_1:	?$
	unknown_2:@?$
	unknown_3:@?
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_252742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1
?
?
while_cond_29868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_29868___redundant_placeholder03
/while_while_cond_29868___redundant_placeholder13
/while_while_cond_29868___redundant_placeholder23
/while_while_cond_29868___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_27407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_27407___redundant_placeholder03
/while_while_cond_27407___redundant_placeholder13
/while_while_cond_27407___redundant_placeholder23
/while_while_cond_27407___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?

?
%__inference_model_layer_call_fn_27612
input_1"
unknown:?$
	unknown_0:@?
	unknown_1:	?$
	unknown_2:@?$
	unknown_3:@?
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_275802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1
?
?
while_cond_27165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_27165___redundant_placeholder03
/while_while_cond_27165___redundant_placeholder13
/while_while_cond_27165___redundant_placeholder23
/while_while_cond_27165___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
conv_lst_m2d_1_while_cond_28456:
6conv_lst_m2d_1_while_conv_lst_m2d_1_while_loop_counter@
<conv_lst_m2d_1_while_conv_lst_m2d_1_while_maximum_iterations$
 conv_lst_m2d_1_while_placeholder&
"conv_lst_m2d_1_while_placeholder_1&
"conv_lst_m2d_1_while_placeholder_2&
"conv_lst_m2d_1_while_placeholder_3:
6conv_lst_m2d_1_while_less_conv_lst_m2d_1_strided_sliceQ
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28456___redundant_placeholder0Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28456___redundant_placeholder1Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28456___redundant_placeholder2Q
Mconv_lst_m2d_1_while_conv_lst_m2d_1_while_cond_28456___redundant_placeholder3!
conv_lst_m2d_1_while_identity
?
conv_lst_m2d_1/while/LessLess conv_lst_m2d_1_while_placeholder6conv_lst_m2d_1_while_less_conv_lst_m2d_1_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d_1/while/Less?
conv_lst_m2d_1/while/IdentityIdentityconv_lst_m2d_1/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d_1/while/Identity"G
conv_lst_m2d_1_while_identity&conv_lst_m2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?9
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_26346

inputs"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_262142
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26278*
condR
while_cond_26277*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityp
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?n
?
!__inference__traced_restore_30990
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: A
&assignvariableop_5_conv_lst_m2d_kernel:?K
0assignvariableop_6_conv_lst_m2d_recurrent_kernel:@?3
$assignvariableop_7_conv_lst_m2d_bias:	?C
(assignvariableop_8_conv_lst_m2d_1_kernel:@?M
2assignvariableop_9_conv_lst_m2d_1_recurrent_kernel:@?6
'assignvariableop_10_conv_lst_m2d_1_bias:	?#
assignvariableop_11_total: #
assignvariableop_12_count: I
.assignvariableop_13_adam_conv_lst_m2d_kernel_m:?S
8assignvariableop_14_adam_conv_lst_m2d_recurrent_kernel_m:@?;
,assignvariableop_15_adam_conv_lst_m2d_bias_m:	?K
0assignvariableop_16_adam_conv_lst_m2d_1_kernel_m:@?U
:assignvariableop_17_adam_conv_lst_m2d_1_recurrent_kernel_m:@?=
.assignvariableop_18_adam_conv_lst_m2d_1_bias_m:	?I
.assignvariableop_19_adam_conv_lst_m2d_kernel_v:?S
8assignvariableop_20_adam_conv_lst_m2d_recurrent_kernel_v:@?;
,assignvariableop_21_adam_conv_lst_m2d_bias_v:	?K
0assignvariableop_22_adam_conv_lst_m2d_1_kernel_v:@?U
:assignvariableop_23_adam_conv_lst_m2d_1_recurrent_kernel_v:@?=
.assignvariableop_24_adam_conv_lst_m2d_1_bias_v:	?
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_conv_lst_m2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp0assignvariableop_6_conv_lst_m2d_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_conv_lst_m2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_conv_lst_m2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp2assignvariableop_9_conv_lst_m2d_1_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_conv_lst_m2d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_adam_conv_lst_m2d_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_conv_lst_m2d_recurrent_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_conv_lst_m2d_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_conv_lst_m2d_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_adam_conv_lst_m2d_1_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_conv_lst_m2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_conv_lst_m2d_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_conv_lst_m2d_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_conv_lst_m2d_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_conv_lst_m2d_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_adam_conv_lst_m2d_1_recurrent_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_conv_lst_m2d_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25f
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_26?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
conv_lst_m2d_while_cond_282386
2conv_lst_m2d_while_conv_lst_m2d_while_loop_counter<
8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations"
conv_lst_m2d_while_placeholder$
 conv_lst_m2d_while_placeholder_1$
 conv_lst_m2d_while_placeholder_2$
 conv_lst_m2d_while_placeholder_36
2conv_lst_m2d_while_less_conv_lst_m2d_strided_sliceM
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_28238___redundant_placeholder0M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_28238___redundant_placeholder1M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_28238___redundant_placeholder2M
Iconv_lst_m2d_while_conv_lst_m2d_while_cond_28238___redundant_placeholder3
conv_lst_m2d_while_identity
?
conv_lst_m2d/while/LessLessconv_lst_m2d_while_placeholder2conv_lst_m2d_while_less_conv_lst_m2d_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Less?
conv_lst_m2d/while/IdentityIdentityconv_lst_m2d/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d/while/Identity"C
conv_lst_m2d_while_identity$conv_lst_m2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?f
?
while_body_28941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_30090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_30090___redundant_placeholder03
/while_while_cond_30090___redundant_placeholder13
/while_while_cond_30090___redundant_placeholder23
/while_while_cond_30090___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_25389
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_25389___redundant_placeholder03
/while_while_cond_25389___redundant_placeholder13
/while_while_cond_25389___redundant_placeholder23
/while_while_cond_25389___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?f
?
while_body_30091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_25390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_25414_0:?(
while_25416_0:@?
while_25418_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_25414:?&
while_25416:@?
while_25418:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_25414_0while_25416_0while_25418_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_253762
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5z

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"
while_25414while_25414_0"
while_25416while_25416_0"
while_25418while_25418_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?	
?
.__inference_conv_lst_m2d_1_layer_call_fn_29518
inputs_0"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_261082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
>
_output_shapes,
*:(????????????????????@
"
_user_specified_name
inputs/0
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_26804

inputs8
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26678*
condR
while_cond_26677*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_26026

inputs

states
states_18
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates
?
?
while_cond_25625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_25625___redundant_placeholder03
/while_while_cond_25625___redundant_placeholder13
/while_while_cond_25625___redundant_placeholder23
/while_while_cond_25625___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?r
?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29995
inputs_08
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilex

zeros_like	ZerosLikeinputs_0*
T0*>
_output_shapes,
*:(????????????????????@2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????@2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:@@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_29869*
condR
while_cond_29868*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:h d
>
_output_shapes,
*:(????????????????????@
"
_user_specified_name
inputs/0
?7
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_25694

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_255642
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_25626*
condR
while_cond_25625*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityp
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?
?
%model_conv_lst_m2d_1_while_cond_25147F
Bmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_loop_counterL
Hmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_maximum_iterations*
&model_conv_lst_m2d_1_while_placeholder,
(model_conv_lst_m2d_1_while_placeholder_1,
(model_conv_lst_m2d_1_while_placeholder_2,
(model_conv_lst_m2d_1_while_placeholder_3F
Bmodel_conv_lst_m2d_1_while_less_model_conv_lst_m2d_1_strided_slice]
Ymodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_cond_25147___redundant_placeholder0]
Ymodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_cond_25147___redundant_placeholder1]
Ymodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_cond_25147___redundant_placeholder2]
Ymodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_cond_25147___redundant_placeholder3'
#model_conv_lst_m2d_1_while_identity
?
model/conv_lst_m2d_1/while/LessLess&model_conv_lst_m2d_1_while_placeholderBmodel_conv_lst_m2d_1_while_less_model_conv_lst_m2d_1_strided_slice*
T0*
_output_shapes
: 2!
model/conv_lst_m2d_1/while/Less?
#model/conv_lst_m2d_1/while/IdentityIdentity#model/conv_lst_m2d_1/while/Less:z:0*
T0
*
_output_shapes
: 2%
#model/conv_lst_m2d_1/while/Identity"S
#model_conv_lst_m2d_1_while_identity,model/conv_lst_m2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?	
?
,__inference_conv_lst_m2d_layer_call_fn_28605
inputs_0"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_256942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
>
_output_shapes,
*:(????????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_28940
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_28940___redundant_placeholder03
/while_while_cond_28940___redundant_placeholder13
/while_while_cond_28940___redundant_placeholder23
/while_while_cond_28940___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
??
?
#model_conv_lst_m2d_while_body_24930B
>model_conv_lst_m2d_while_model_conv_lst_m2d_while_loop_counterH
Dmodel_conv_lst_m2d_while_model_conv_lst_m2d_while_maximum_iterations(
$model_conv_lst_m2d_while_placeholder*
&model_conv_lst_m2d_while_placeholder_1*
&model_conv_lst_m2d_while_placeholder_2*
&model_conv_lst_m2d_while_placeholder_3?
;model_conv_lst_m2d_while_model_conv_lst_m2d_strided_slice_0}
ymodel_conv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0S
8model_conv_lst_m2d_while_split_readvariableop_resource_0:?U
:model_conv_lst_m2d_while_split_1_readvariableop_resource_0:@?I
:model_conv_lst_m2d_while_split_2_readvariableop_resource_0:	?%
!model_conv_lst_m2d_while_identity'
#model_conv_lst_m2d_while_identity_1'
#model_conv_lst_m2d_while_identity_2'
#model_conv_lst_m2d_while_identity_3'
#model_conv_lst_m2d_while_identity_4'
#model_conv_lst_m2d_while_identity_5=
9model_conv_lst_m2d_while_model_conv_lst_m2d_strided_slice{
wmodel_conv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensorQ
6model_conv_lst_m2d_while_split_readvariableop_resource:?S
8model_conv_lst_m2d_while_split_1_readvariableop_resource:@?G
8model_conv_lst_m2d_while_split_2_readvariableop_resource:	???-model/conv_lst_m2d/while/split/ReadVariableOp?/model/conv_lst_m2d/while/split_1/ReadVariableOp?/model/conv_lst_m2d/while/split_2/ReadVariableOp?
Jmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2L
Jmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<model/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemymodel_conv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0$model_conv_lst_m2d_while_placeholderSmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02>
<model/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem?
(model/conv_lst_m2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/conv_lst_m2d/while/split/split_dim?
-model/conv_lst_m2d/while/split/ReadVariableOpReadVariableOp8model_conv_lst_m2d_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02/
-model/conv_lst_m2d/while/split/ReadVariableOp?
model/conv_lst_m2d/while/splitSplit1model/conv_lst_m2d/while/split/split_dim:output:05model/conv_lst_m2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2 
model/conv_lst_m2d/while/split?
*model/conv_lst_m2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model/conv_lst_m2d/while/split_1/split_dim?
/model/conv_lst_m2d/while/split_1/ReadVariableOpReadVariableOp:model_conv_lst_m2d_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype021
/model/conv_lst_m2d/while/split_1/ReadVariableOp?
 model/conv_lst_m2d/while/split_1Split3model/conv_lst_m2d/while/split_1/split_dim:output:07model/conv_lst_m2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2"
 model/conv_lst_m2d/while/split_1?
*model/conv_lst_m2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model/conv_lst_m2d/while/split_2/split_dim?
/model/conv_lst_m2d/while/split_2/ReadVariableOpReadVariableOp:model_conv_lst_m2d_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/model/conv_lst_m2d/while/split_2/ReadVariableOp?
 model/conv_lst_m2d/while/split_2Split3model/conv_lst_m2d/while/split_2/split_dim:output:07model/conv_lst_m2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2"
 model/conv_lst_m2d/while/split_2?
$model/conv_lst_m2d/while/convolutionConv2DCmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0'model/conv_lst_m2d/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2&
$model/conv_lst_m2d/while/convolution?
 model/conv_lst_m2d/while/BiasAddBiasAdd-model/conv_lst_m2d/while/convolution:output:0)model/conv_lst_m2d/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d/while/BiasAdd?
&model/conv_lst_m2d/while/convolution_1Conv2DCmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0'model/conv_lst_m2d/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_1?
"model/conv_lst_m2d/while/BiasAdd_1BiasAdd/model/conv_lst_m2d/while/convolution_1:output:0)model/conv_lst_m2d/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d/while/BiasAdd_1?
&model/conv_lst_m2d/while/convolution_2Conv2DCmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0'model/conv_lst_m2d/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_2?
"model/conv_lst_m2d/while/BiasAdd_2BiasAdd/model/conv_lst_m2d/while/convolution_2:output:0)model/conv_lst_m2d/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d/while/BiasAdd_2?
&model/conv_lst_m2d/while/convolution_3Conv2DCmodel/conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0'model/conv_lst_m2d/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_3?
"model/conv_lst_m2d/while/BiasAdd_3BiasAdd/model/conv_lst_m2d/while/convolution_3:output:0)model/conv_lst_m2d/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d/while/BiasAdd_3?
&model/conv_lst_m2d/while/convolution_4Conv2D&model_conv_lst_m2d_while_placeholder_2)model/conv_lst_m2d/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_4?
&model/conv_lst_m2d/while/convolution_5Conv2D&model_conv_lst_m2d_while_placeholder_2)model/conv_lst_m2d/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_5?
&model/conv_lst_m2d/while/convolution_6Conv2D&model_conv_lst_m2d_while_placeholder_2)model/conv_lst_m2d/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_6?
&model/conv_lst_m2d/while/convolution_7Conv2D&model_conv_lst_m2d_while_placeholder_2)model/conv_lst_m2d/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d/while/convolution_7?
model/conv_lst_m2d/while/addAddV2)model/conv_lst_m2d/while/BiasAdd:output:0/model/conv_lst_m2d/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/while/add?
model/conv_lst_m2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
model/conv_lst_m2d/while/Const?
 model/conv_lst_m2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 model/conv_lst_m2d/while/Const_1?
model/conv_lst_m2d/while/MulMul model/conv_lst_m2d/while/add:z:0'model/conv_lst_m2d/while/Const:output:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/while/Mul?
model/conv_lst_m2d/while/Add_1AddV2 model/conv_lst_m2d/while/Mul:z:0)model/conv_lst_m2d/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/Add_1?
0model/conv_lst_m2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0model/conv_lst_m2d/while/clip_by_value/Minimum/y?
.model/conv_lst_m2d/while/clip_by_value/MinimumMinimum"model/conv_lst_m2d/while/Add_1:z:09model/conv_lst_m2d/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@20
.model/conv_lst_m2d/while/clip_by_value/Minimum?
(model/conv_lst_m2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(model/conv_lst_m2d/while/clip_by_value/y?
&model/conv_lst_m2d/while/clip_by_valueMaximum2model/conv_lst_m2d/while/clip_by_value/Minimum:z:01model/conv_lst_m2d/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2(
&model/conv_lst_m2d/while/clip_by_value?
model/conv_lst_m2d/while/add_2AddV2+model/conv_lst_m2d/while/BiasAdd_1:output:0/model/conv_lst_m2d/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/add_2?
 model/conv_lst_m2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model/conv_lst_m2d/while/Const_2?
 model/conv_lst_m2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 model/conv_lst_m2d/while/Const_3?
model/conv_lst_m2d/while/Mul_1Mul"model/conv_lst_m2d/while/add_2:z:0)model/conv_lst_m2d/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/Mul_1?
model/conv_lst_m2d/while/Add_3AddV2"model/conv_lst_m2d/while/Mul_1:z:0)model/conv_lst_m2d/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/Add_3?
2model/conv_lst_m2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2model/conv_lst_m2d/while/clip_by_value_1/Minimum/y?
0model/conv_lst_m2d/while/clip_by_value_1/MinimumMinimum"model/conv_lst_m2d/while/Add_3:z:0;model/conv_lst_m2d/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@22
0model/conv_lst_m2d/while/clip_by_value_1/Minimum?
*model/conv_lst_m2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model/conv_lst_m2d/while/clip_by_value_1/y?
(model/conv_lst_m2d/while/clip_by_value_1Maximum4model/conv_lst_m2d/while/clip_by_value_1/Minimum:z:03model/conv_lst_m2d/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2*
(model/conv_lst_m2d/while/clip_by_value_1?
model/conv_lst_m2d/while/mul_2Mul,model/conv_lst_m2d/while/clip_by_value_1:z:0&model_conv_lst_m2d_while_placeholder_3*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/mul_2?
model/conv_lst_m2d/while/add_4AddV2+model/conv_lst_m2d/while/BiasAdd_2:output:0/model/conv_lst_m2d/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/add_4?
model/conv_lst_m2d/while/ReluRelu"model/conv_lst_m2d/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2
model/conv_lst_m2d/while/Relu?
model/conv_lst_m2d/while/mul_3Mul*model/conv_lst_m2d/while/clip_by_value:z:0+model/conv_lst_m2d/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/mul_3?
model/conv_lst_m2d/while/add_5AddV2"model/conv_lst_m2d/while/mul_2:z:0"model/conv_lst_m2d/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/add_5?
model/conv_lst_m2d/while/add_6AddV2+model/conv_lst_m2d/while/BiasAdd_3:output:0/model/conv_lst_m2d/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/add_6?
 model/conv_lst_m2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model/conv_lst_m2d/while/Const_4?
 model/conv_lst_m2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 model/conv_lst_m2d/while/Const_5?
model/conv_lst_m2d/while/Mul_4Mul"model/conv_lst_m2d/while/add_6:z:0)model/conv_lst_m2d/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/Mul_4?
model/conv_lst_m2d/while/Add_7AddV2"model/conv_lst_m2d/while/Mul_4:z:0)model/conv_lst_m2d/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/Add_7?
2model/conv_lst_m2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2model/conv_lst_m2d/while/clip_by_value_2/Minimum/y?
0model/conv_lst_m2d/while/clip_by_value_2/MinimumMinimum"model/conv_lst_m2d/while/Add_7:z:0;model/conv_lst_m2d/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@22
0model/conv_lst_m2d/while/clip_by_value_2/Minimum?
*model/conv_lst_m2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model/conv_lst_m2d/while/clip_by_value_2/y?
(model/conv_lst_m2d/while/clip_by_value_2Maximum4model/conv_lst_m2d/while/clip_by_value_2/Minimum:z:03model/conv_lst_m2d/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2*
(model/conv_lst_m2d/while/clip_by_value_2?
model/conv_lst_m2d/while/Relu_1Relu"model/conv_lst_m2d/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2!
model/conv_lst_m2d/while/Relu_1?
model/conv_lst_m2d/while/mul_5Mul,model/conv_lst_m2d/while/clip_by_value_2:z:0-model/conv_lst_m2d/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d/while/mul_5?
=model/conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&model_conv_lst_m2d_while_placeholder_1$model_conv_lst_m2d_while_placeholder"model/conv_lst_m2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype02?
=model/conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItem?
 model/conv_lst_m2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv_lst_m2d/while/add_8/y?
model/conv_lst_m2d/while/add_8AddV2$model_conv_lst_m2d_while_placeholder)model/conv_lst_m2d/while/add_8/y:output:0*
T0*
_output_shapes
: 2 
model/conv_lst_m2d/while/add_8?
 model/conv_lst_m2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv_lst_m2d/while/add_9/y?
model/conv_lst_m2d/while/add_9AddV2>model_conv_lst_m2d_while_model_conv_lst_m2d_while_loop_counter)model/conv_lst_m2d/while/add_9/y:output:0*
T0*
_output_shapes
: 2 
model/conv_lst_m2d/while/add_9?
!model/conv_lst_m2d/while/IdentityIdentity"model/conv_lst_m2d/while/add_9:z:0^model/conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2#
!model/conv_lst_m2d/while/Identity?
#model/conv_lst_m2d/while/Identity_1IdentityDmodel_conv_lst_m2d_while_model_conv_lst_m2d_while_maximum_iterations^model/conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2%
#model/conv_lst_m2d/while/Identity_1?
#model/conv_lst_m2d/while/Identity_2Identity"model/conv_lst_m2d/while/add_8:z:0^model/conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2%
#model/conv_lst_m2d/while/Identity_2?
#model/conv_lst_m2d/while/Identity_3IdentityMmodel/conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2%
#model/conv_lst_m2d/while/Identity_3?
#model/conv_lst_m2d/while/Identity_4Identity"model/conv_lst_m2d/while/mul_5:z:0^model/conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2%
#model/conv_lst_m2d/while/Identity_4?
#model/conv_lst_m2d/while/Identity_5Identity"model/conv_lst_m2d/while/add_5:z:0^model/conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2%
#model/conv_lst_m2d/while/Identity_5?
model/conv_lst_m2d/while/NoOpNoOp.^model/conv_lst_m2d/while/split/ReadVariableOp0^model/conv_lst_m2d/while/split_1/ReadVariableOp0^model/conv_lst_m2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
model/conv_lst_m2d/while/NoOp"O
!model_conv_lst_m2d_while_identity*model/conv_lst_m2d/while/Identity:output:0"S
#model_conv_lst_m2d_while_identity_1,model/conv_lst_m2d/while/Identity_1:output:0"S
#model_conv_lst_m2d_while_identity_2,model/conv_lst_m2d/while/Identity_2:output:0"S
#model_conv_lst_m2d_while_identity_3,model/conv_lst_m2d/while/Identity_3:output:0"S
#model_conv_lst_m2d_while_identity_4,model/conv_lst_m2d/while/Identity_4:output:0"S
#model_conv_lst_m2d_while_identity_5,model/conv_lst_m2d/while/Identity_5:output:0"x
9model_conv_lst_m2d_while_model_conv_lst_m2d_strided_slice;model_conv_lst_m2d_while_model_conv_lst_m2d_strided_slice_0"v
8model_conv_lst_m2d_while_split_1_readvariableop_resource:model_conv_lst_m2d_while_split_1_readvariableop_resource_0"v
8model_conv_lst_m2d_while_split_2_readvariableop_resource:model_conv_lst_m2d_while_split_2_readvariableop_resource_0"r
6model_conv_lst_m2d_while_split_readvariableop_resource8model_conv_lst_m2d_while_split_readvariableop_resource_0"?
wmodel_conv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensorymodel_conv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2^
-model/conv_lst_m2d/while/split/ReadVariableOp-model/conv_lst_m2d/while/split/ReadVariableOp2b
/model/conv_lst_m2d/while/split_1/ReadVariableOp/model/conv_lst_m2d/while/split_1/ReadVariableOp2b
/model/conv_lst_m2d/while/split_2/ReadVariableOp/model/conv_lst_m2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
Ѕ
?
conv_lst_m2d_while_body_278016
2conv_lst_m2d_while_conv_lst_m2d_while_loop_counter<
8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations"
conv_lst_m2d_while_placeholder$
 conv_lst_m2d_while_placeholder_1$
 conv_lst_m2d_while_placeholder_2$
 conv_lst_m2d_while_placeholder_33
/conv_lst_m2d_while_conv_lst_m2d_strided_slice_0q
mconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0M
2conv_lst_m2d_while_split_readvariableop_resource_0:?O
4conv_lst_m2d_while_split_1_readvariableop_resource_0:@?C
4conv_lst_m2d_while_split_2_readvariableop_resource_0:	?
conv_lst_m2d_while_identity!
conv_lst_m2d_while_identity_1!
conv_lst_m2d_while_identity_2!
conv_lst_m2d_while_identity_3!
conv_lst_m2d_while_identity_4!
conv_lst_m2d_while_identity_51
-conv_lst_m2d_while_conv_lst_m2d_strided_sliceo
kconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensorK
0conv_lst_m2d_while_split_readvariableop_resource:?M
2conv_lst_m2d_while_split_1_readvariableop_resource:@?A
2conv_lst_m2d_while_split_2_readvariableop_resource:	???'conv_lst_m2d/while/split/ReadVariableOp?)conv_lst_m2d/while/split_1/ReadVariableOp?)conv_lst_m2d/while/split_2/ReadVariableOp?
Dconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2F
Dconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0conv_lst_m2d_while_placeholderMconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype028
6conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem?
"conv_lst_m2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"conv_lst_m2d/while/split/split_dim?
'conv_lst_m2d/while/split/ReadVariableOpReadVariableOp2conv_lst_m2d_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02)
'conv_lst_m2d/while/split/ReadVariableOp?
conv_lst_m2d/while/splitSplit+conv_lst_m2d/while/split/split_dim:output:0/conv_lst_m2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
conv_lst_m2d/while/split?
$conv_lst_m2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d/while/split_1/split_dim?
)conv_lst_m2d/while/split_1/ReadVariableOpReadVariableOp4conv_lst_m2d_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02+
)conv_lst_m2d/while/split_1/ReadVariableOp?
conv_lst_m2d/while/split_1Split-conv_lst_m2d/while/split_1/split_dim:output:01conv_lst_m2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d/while/split_1?
$conv_lst_m2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$conv_lst_m2d/while/split_2/split_dim?
)conv_lst_m2d/while/split_2/ReadVariableOpReadVariableOp4conv_lst_m2d_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)conv_lst_m2d/while/split_2/ReadVariableOp?
conv_lst_m2d/while/split_2Split-conv_lst_m2d/while/split_2/split_dim:output:01conv_lst_m2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d/while/split_2?
conv_lst_m2d/while/convolutionConv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2 
conv_lst_m2d/while/convolution?
conv_lst_m2d/while/BiasAddBiasAdd'conv_lst_m2d/while/convolution:output:0#conv_lst_m2d/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd?
 conv_lst_m2d/while/convolution_1Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_1?
conv_lst_m2d/while/BiasAdd_1BiasAdd)conv_lst_m2d/while/convolution_1:output:0#conv_lst_m2d/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_1?
 conv_lst_m2d/while/convolution_2Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_2?
conv_lst_m2d/while/BiasAdd_2BiasAdd)conv_lst_m2d/while/convolution_2:output:0#conv_lst_m2d/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_2?
 conv_lst_m2d/while/convolution_3Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_3?
conv_lst_m2d/while/BiasAdd_3BiasAdd)conv_lst_m2d/while/convolution_3:output:0#conv_lst_m2d/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_3?
 conv_lst_m2d/while/convolution_4Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_4?
 conv_lst_m2d/while/convolution_5Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_5?
 conv_lst_m2d/while/convolution_6Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_6?
 conv_lst_m2d/while/convolution_7Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_7?
conv_lst_m2d/while/addAddV2#conv_lst_m2d/while/BiasAdd:output:0)conv_lst_m2d/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/addy
conv_lst_m2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const}
conv_lst_m2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_1?
conv_lst_m2d/while/MulMulconv_lst_m2d/while/add:z:0!conv_lst_m2d/while/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul?
conv_lst_m2d/while/Add_1AddV2conv_lst_m2d/while/Mul:z:0#conv_lst_m2d/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_1?
*conv_lst_m2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*conv_lst_m2d/while/clip_by_value/Minimum/y?
(conv_lst_m2d/while/clip_by_value/MinimumMinimumconv_lst_m2d/while/Add_1:z:03conv_lst_m2d/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2*
(conv_lst_m2d/while/clip_by_value/Minimum?
"conv_lst_m2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv_lst_m2d/while/clip_by_value/y?
 conv_lst_m2d/while/clip_by_valueMaximum,conv_lst_m2d/while/clip_by_value/Minimum:z:0+conv_lst_m2d/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2"
 conv_lst_m2d/while/clip_by_value?
conv_lst_m2d/while/add_2AddV2%conv_lst_m2d/while/BiasAdd_1:output:0)conv_lst_m2d/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_2}
conv_lst_m2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const_2}
conv_lst_m2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_3?
conv_lst_m2d/while/Mul_1Mulconv_lst_m2d/while/add_2:z:0#conv_lst_m2d/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul_1?
conv_lst_m2d/while/Add_3AddV2conv_lst_m2d/while/Mul_1:z:0#conv_lst_m2d/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_3?
,conv_lst_m2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d/while/clip_by_value_1/Minimum/y?
*conv_lst_m2d/while/clip_by_value_1/MinimumMinimumconv_lst_m2d/while/Add_3:z:05conv_lst_m2d/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d/while/clip_by_value_1/Minimum?
$conv_lst_m2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d/while/clip_by_value_1/y?
"conv_lst_m2d/while/clip_by_value_1Maximum.conv_lst_m2d/while/clip_by_value_1/Minimum:z:0-conv_lst_m2d/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/while/clip_by_value_1?
conv_lst_m2d/while/mul_2Mul&conv_lst_m2d/while/clip_by_value_1:z:0 conv_lst_m2d_while_placeholder_3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_2?
conv_lst_m2d/while/add_4AddV2%conv_lst_m2d/while/BiasAdd_2:output:0)conv_lst_m2d/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_4?
conv_lst_m2d/while/ReluReluconv_lst_m2d/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Relu?
conv_lst_m2d/while/mul_3Mul$conv_lst_m2d/while/clip_by_value:z:0%conv_lst_m2d/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_3?
conv_lst_m2d/while/add_5AddV2conv_lst_m2d/while/mul_2:z:0conv_lst_m2d/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_5?
conv_lst_m2d/while/add_6AddV2%conv_lst_m2d/while/BiasAdd_3:output:0)conv_lst_m2d/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_6}
conv_lst_m2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const_4}
conv_lst_m2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_5?
conv_lst_m2d/while/Mul_4Mulconv_lst_m2d/while/add_6:z:0#conv_lst_m2d/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul_4?
conv_lst_m2d/while/Add_7AddV2conv_lst_m2d/while/Mul_4:z:0#conv_lst_m2d/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_7?
,conv_lst_m2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d/while/clip_by_value_2/Minimum/y?
*conv_lst_m2d/while/clip_by_value_2/MinimumMinimumconv_lst_m2d/while/Add_7:z:05conv_lst_m2d/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d/while/clip_by_value_2/Minimum?
$conv_lst_m2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d/while/clip_by_value_2/y?
"conv_lst_m2d/while/clip_by_value_2Maximum.conv_lst_m2d/while/clip_by_value_2/Minimum:z:0-conv_lst_m2d/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/while/clip_by_value_2?
conv_lst_m2d/while/Relu_1Reluconv_lst_m2d/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Relu_1?
conv_lst_m2d/while/mul_5Mul&conv_lst_m2d/while/clip_by_value_2:z:0'conv_lst_m2d/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_5?
7conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem conv_lst_m2d_while_placeholder_1conv_lst_m2d_while_placeholderconv_lst_m2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype029
7conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItemz
conv_lst_m2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/while/add_8/y?
conv_lst_m2d/while/add_8AddV2conv_lst_m2d_while_placeholder#conv_lst_m2d/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d/while/add_8z
conv_lst_m2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/while/add_9/y?
conv_lst_m2d/while/add_9AddV22conv_lst_m2d_while_conv_lst_m2d_while_loop_counter#conv_lst_m2d/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d/while/add_9?
conv_lst_m2d/while/IdentityIdentityconv_lst_m2d/while/add_9:z:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity?
conv_lst_m2d/while/Identity_1Identity8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_1?
conv_lst_m2d/while/Identity_2Identityconv_lst_m2d/while/add_8:z:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_2?
conv_lst_m2d/while/Identity_3IdentityGconv_lst_m2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_3?
conv_lst_m2d/while/Identity_4Identityconv_lst_m2d/while/mul_5:z:0^conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Identity_4?
conv_lst_m2d/while/Identity_5Identityconv_lst_m2d/while/add_5:z:0^conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Identity_5?
conv_lst_m2d/while/NoOpNoOp(^conv_lst_m2d/while/split/ReadVariableOp*^conv_lst_m2d/while/split_1/ReadVariableOp*^conv_lst_m2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
conv_lst_m2d/while/NoOp"`
-conv_lst_m2d_while_conv_lst_m2d_strided_slice/conv_lst_m2d_while_conv_lst_m2d_strided_slice_0"C
conv_lst_m2d_while_identity$conv_lst_m2d/while/Identity:output:0"G
conv_lst_m2d_while_identity_1&conv_lst_m2d/while/Identity_1:output:0"G
conv_lst_m2d_while_identity_2&conv_lst_m2d/while/Identity_2:output:0"G
conv_lst_m2d_while_identity_3&conv_lst_m2d/while/Identity_3:output:0"G
conv_lst_m2d_while_identity_4&conv_lst_m2d/while/Identity_4:output:0"G
conv_lst_m2d_while_identity_5&conv_lst_m2d/while/Identity_5:output:0"j
2conv_lst_m2d_while_split_1_readvariableop_resource4conv_lst_m2d_while_split_1_readvariableop_resource_0"j
2conv_lst_m2d_while_split_2_readvariableop_resource4conv_lst_m2d_while_split_2_readvariableop_resource_0"f
0conv_lst_m2d_while_split_readvariableop_resource2conv_lst_m2d_while_split_readvariableop_resource_0"?
kconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensormconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2R
'conv_lst_m2d/while/split/ReadVariableOp'conv_lst_m2d/while/split/ReadVariableOp2V
)conv_lst_m2d/while/split_1/ReadVariableOp)conv_lst_m2d/while/split_1/ReadVariableOp2V
)conv_lst_m2d/while/split_2/ReadVariableOp)conv_lst_m2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?E
?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_25376

inputs

states
states_18
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates:YU
1
_output_shapes
:???????????@
 
_user_specified_namestates
??
?
%model_conv_lst_m2d_1_while_body_25148F
Bmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_loop_counterL
Hmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_maximum_iterations*
&model_conv_lst_m2d_1_while_placeholder,
(model_conv_lst_m2d_1_while_placeholder_1,
(model_conv_lst_m2d_1_while_placeholder_2,
(model_conv_lst_m2d_1_while_placeholder_3C
?model_conv_lst_m2d_1_while_model_conv_lst_m2d_1_strided_slice_0?
}model_conv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0U
:model_conv_lst_m2d_1_while_split_readvariableop_resource_0:@?W
<model_conv_lst_m2d_1_while_split_1_readvariableop_resource_0:@?K
<model_conv_lst_m2d_1_while_split_2_readvariableop_resource_0:	?'
#model_conv_lst_m2d_1_while_identity)
%model_conv_lst_m2d_1_while_identity_1)
%model_conv_lst_m2d_1_while_identity_2)
%model_conv_lst_m2d_1_while_identity_3)
%model_conv_lst_m2d_1_while_identity_4)
%model_conv_lst_m2d_1_while_identity_5A
=model_conv_lst_m2d_1_while_model_conv_lst_m2d_1_strided_slice
{model_conv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensorS
8model_conv_lst_m2d_1_while_split_readvariableop_resource:@?U
:model_conv_lst_m2d_1_while_split_1_readvariableop_resource:@?I
:model_conv_lst_m2d_1_while_split_2_readvariableop_resource:	???/model/conv_lst_m2d_1/while/split/ReadVariableOp?1model/conv_lst_m2d_1/while/split_1/ReadVariableOp?1model/conv_lst_m2d_1/while/split_2/ReadVariableOp?
Lmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2N
Lmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>model/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_conv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0&model_conv_lst_m2d_1_while_placeholderUmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02@
>model/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem?
*model/conv_lst_m2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model/conv_lst_m2d_1/while/split/split_dim?
/model/conv_lst_m2d_1/while/split/ReadVariableOpReadVariableOp:model_conv_lst_m2d_1_while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype021
/model/conv_lst_m2d_1/while/split/ReadVariableOp?
 model/conv_lst_m2d_1/while/splitSplit3model/conv_lst_m2d_1/while/split/split_dim:output:07model/conv_lst_m2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2"
 model/conv_lst_m2d_1/while/split?
,model/conv_lst_m2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model/conv_lst_m2d_1/while/split_1/split_dim?
1model/conv_lst_m2d_1/while/split_1/ReadVariableOpReadVariableOp<model_conv_lst_m2d_1_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype023
1model/conv_lst_m2d_1/while/split_1/ReadVariableOp?
"model/conv_lst_m2d_1/while/split_1Split5model/conv_lst_m2d_1/while/split_1/split_dim:output:09model/conv_lst_m2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2$
"model/conv_lst_m2d_1/while/split_1?
,model/conv_lst_m2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/conv_lst_m2d_1/while/split_2/split_dim?
1model/conv_lst_m2d_1/while/split_2/ReadVariableOpReadVariableOp<model_conv_lst_m2d_1_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1model/conv_lst_m2d_1/while/split_2/ReadVariableOp?
"model/conv_lst_m2d_1/while/split_2Split5model/conv_lst_m2d_1/while/split_2/split_dim:output:09model/conv_lst_m2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2$
"model/conv_lst_m2d_1/while/split_2?
&model/conv_lst_m2d_1/while/convolutionConv2DEmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0)model/conv_lst_m2d_1/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2(
&model/conv_lst_m2d_1/while/convolution?
"model/conv_lst_m2d_1/while/BiasAddBiasAdd/model/conv_lst_m2d_1/while/convolution:output:0+model/conv_lst_m2d_1/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2$
"model/conv_lst_m2d_1/while/BiasAdd?
(model/conv_lst_m2d_1/while/convolution_1Conv2DEmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0)model/conv_lst_m2d_1/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_1?
$model/conv_lst_m2d_1/while/BiasAdd_1BiasAdd1model/conv_lst_m2d_1/while/convolution_1:output:0+model/conv_lst_m2d_1/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2&
$model/conv_lst_m2d_1/while/BiasAdd_1?
(model/conv_lst_m2d_1/while/convolution_2Conv2DEmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0)model/conv_lst_m2d_1/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_2?
$model/conv_lst_m2d_1/while/BiasAdd_2BiasAdd1model/conv_lst_m2d_1/while/convolution_2:output:0+model/conv_lst_m2d_1/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2&
$model/conv_lst_m2d_1/while/BiasAdd_2?
(model/conv_lst_m2d_1/while/convolution_3Conv2DEmodel/conv_lst_m2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0)model/conv_lst_m2d_1/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_3?
$model/conv_lst_m2d_1/while/BiasAdd_3BiasAdd1model/conv_lst_m2d_1/while/convolution_3:output:0+model/conv_lst_m2d_1/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2&
$model/conv_lst_m2d_1/while/BiasAdd_3?
(model/conv_lst_m2d_1/while/convolution_4Conv2D(model_conv_lst_m2d_1_while_placeholder_2+model/conv_lst_m2d_1/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_4?
(model/conv_lst_m2d_1/while/convolution_5Conv2D(model_conv_lst_m2d_1_while_placeholder_2+model/conv_lst_m2d_1/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_5?
(model/conv_lst_m2d_1/while/convolution_6Conv2D(model_conv_lst_m2d_1_while_placeholder_2+model/conv_lst_m2d_1/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_6?
(model/conv_lst_m2d_1/while/convolution_7Conv2D(model_conv_lst_m2d_1_while_placeholder_2+model/conv_lst_m2d_1/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2*
(model/conv_lst_m2d_1/while/convolution_7?
model/conv_lst_m2d_1/while/addAddV2+model/conv_lst_m2d_1/while/BiasAdd:output:01model/conv_lst_m2d_1/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d_1/while/add?
 model/conv_lst_m2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model/conv_lst_m2d_1/while/Const?
"model/conv_lst_m2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"model/conv_lst_m2d_1/while/Const_1?
model/conv_lst_m2d_1/while/MulMul"model/conv_lst_m2d_1/while/add:z:0)model/conv_lst_m2d_1/while/Const:output:0*
T0*1
_output_shapes
:???????????@2 
model/conv_lst_m2d_1/while/Mul?
 model/conv_lst_m2d_1/while/Add_1AddV2"model/conv_lst_m2d_1/while/Mul:z:0+model/conv_lst_m2d_1/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/Add_1?
2model/conv_lst_m2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2model/conv_lst_m2d_1/while/clip_by_value/Minimum/y?
0model/conv_lst_m2d_1/while/clip_by_value/MinimumMinimum$model/conv_lst_m2d_1/while/Add_1:z:0;model/conv_lst_m2d_1/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@22
0model/conv_lst_m2d_1/while/clip_by_value/Minimum?
*model/conv_lst_m2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model/conv_lst_m2d_1/while/clip_by_value/y?
(model/conv_lst_m2d_1/while/clip_by_valueMaximum4model/conv_lst_m2d_1/while/clip_by_value/Minimum:z:03model/conv_lst_m2d_1/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2*
(model/conv_lst_m2d_1/while/clip_by_value?
 model/conv_lst_m2d_1/while/add_2AddV2-model/conv_lst_m2d_1/while/BiasAdd_1:output:01model/conv_lst_m2d_1/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/add_2?
"model/conv_lst_m2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"model/conv_lst_m2d_1/while/Const_2?
"model/conv_lst_m2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"model/conv_lst_m2d_1/while/Const_3?
 model/conv_lst_m2d_1/while/Mul_1Mul$model/conv_lst_m2d_1/while/add_2:z:0+model/conv_lst_m2d_1/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/Mul_1?
 model/conv_lst_m2d_1/while/Add_3AddV2$model/conv_lst_m2d_1/while/Mul_1:z:0+model/conv_lst_m2d_1/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/Add_3?
4model/conv_lst_m2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4model/conv_lst_m2d_1/while/clip_by_value_1/Minimum/y?
2model/conv_lst_m2d_1/while/clip_by_value_1/MinimumMinimum$model/conv_lst_m2d_1/while/Add_3:z:0=model/conv_lst_m2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@24
2model/conv_lst_m2d_1/while/clip_by_value_1/Minimum?
,model/conv_lst_m2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model/conv_lst_m2d_1/while/clip_by_value_1/y?
*model/conv_lst_m2d_1/while/clip_by_value_1Maximum6model/conv_lst_m2d_1/while/clip_by_value_1/Minimum:z:05model/conv_lst_m2d_1/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2,
*model/conv_lst_m2d_1/while/clip_by_value_1?
 model/conv_lst_m2d_1/while/mul_2Mul.model/conv_lst_m2d_1/while/clip_by_value_1:z:0(model_conv_lst_m2d_1_while_placeholder_3*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/mul_2?
 model/conv_lst_m2d_1/while/add_4AddV2-model/conv_lst_m2d_1/while/BiasAdd_2:output:01model/conv_lst_m2d_1/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/add_4?
model/conv_lst_m2d_1/while/ReluRelu$model/conv_lst_m2d_1/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2!
model/conv_lst_m2d_1/while/Relu?
 model/conv_lst_m2d_1/while/mul_3Mul,model/conv_lst_m2d_1/while/clip_by_value:z:0-model/conv_lst_m2d_1/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/mul_3?
 model/conv_lst_m2d_1/while/add_5AddV2$model/conv_lst_m2d_1/while/mul_2:z:0$model/conv_lst_m2d_1/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/add_5?
 model/conv_lst_m2d_1/while/add_6AddV2-model/conv_lst_m2d_1/while/BiasAdd_3:output:01model/conv_lst_m2d_1/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/add_6?
"model/conv_lst_m2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"model/conv_lst_m2d_1/while/Const_4?
"model/conv_lst_m2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"model/conv_lst_m2d_1/while/Const_5?
 model/conv_lst_m2d_1/while/Mul_4Mul$model/conv_lst_m2d_1/while/add_6:z:0+model/conv_lst_m2d_1/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/Mul_4?
 model/conv_lst_m2d_1/while/Add_7AddV2$model/conv_lst_m2d_1/while/Mul_4:z:0+model/conv_lst_m2d_1/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/Add_7?
4model/conv_lst_m2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4model/conv_lst_m2d_1/while/clip_by_value_2/Minimum/y?
2model/conv_lst_m2d_1/while/clip_by_value_2/MinimumMinimum$model/conv_lst_m2d_1/while/Add_7:z:0=model/conv_lst_m2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@24
2model/conv_lst_m2d_1/while/clip_by_value_2/Minimum?
,model/conv_lst_m2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model/conv_lst_m2d_1/while/clip_by_value_2/y?
*model/conv_lst_m2d_1/while/clip_by_value_2Maximum6model/conv_lst_m2d_1/while/clip_by_value_2/Minimum:z:05model/conv_lst_m2d_1/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2,
*model/conv_lst_m2d_1/while/clip_by_value_2?
!model/conv_lst_m2d_1/while/Relu_1Relu$model/conv_lst_m2d_1/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2#
!model/conv_lst_m2d_1/while/Relu_1?
 model/conv_lst_m2d_1/while/mul_5Mul.model/conv_lst_m2d_1/while/clip_by_value_2:z:0/model/conv_lst_m2d_1/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2"
 model/conv_lst_m2d_1/while/mul_5?
?model/conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_conv_lst_m2d_1_while_placeholder_1&model_conv_lst_m2d_1_while_placeholder$model/conv_lst_m2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype02A
?model/conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem?
"model/conv_lst_m2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/conv_lst_m2d_1/while/add_8/y?
 model/conv_lst_m2d_1/while/add_8AddV2&model_conv_lst_m2d_1_while_placeholder+model/conv_lst_m2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2"
 model/conv_lst_m2d_1/while/add_8?
"model/conv_lst_m2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/conv_lst_m2d_1/while/add_9/y?
 model/conv_lst_m2d_1/while/add_9AddV2Bmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_loop_counter+model/conv_lst_m2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2"
 model/conv_lst_m2d_1/while/add_9?
#model/conv_lst_m2d_1/while/IdentityIdentity$model/conv_lst_m2d_1/while/add_9:z:0 ^model/conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2%
#model/conv_lst_m2d_1/while/Identity?
%model/conv_lst_m2d_1/while/Identity_1IdentityHmodel_conv_lst_m2d_1_while_model_conv_lst_m2d_1_while_maximum_iterations ^model/conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2'
%model/conv_lst_m2d_1/while/Identity_1?
%model/conv_lst_m2d_1/while/Identity_2Identity$model/conv_lst_m2d_1/while/add_8:z:0 ^model/conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2'
%model/conv_lst_m2d_1/while/Identity_2?
%model/conv_lst_m2d_1/while/Identity_3IdentityOmodel/conv_lst_m2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^model/conv_lst_m2d_1/while/NoOp*
T0*
_output_shapes
: 2'
%model/conv_lst_m2d_1/while/Identity_3?
%model/conv_lst_m2d_1/while/Identity_4Identity$model/conv_lst_m2d_1/while/mul_5:z:0 ^model/conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2'
%model/conv_lst_m2d_1/while/Identity_4?
%model/conv_lst_m2d_1/while/Identity_5Identity$model/conv_lst_m2d_1/while/add_5:z:0 ^model/conv_lst_m2d_1/while/NoOp*
T0*1
_output_shapes
:???????????@2'
%model/conv_lst_m2d_1/while/Identity_5?
model/conv_lst_m2d_1/while/NoOpNoOp0^model/conv_lst_m2d_1/while/split/ReadVariableOp2^model/conv_lst_m2d_1/while/split_1/ReadVariableOp2^model/conv_lst_m2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
model/conv_lst_m2d_1/while/NoOp"S
#model_conv_lst_m2d_1_while_identity,model/conv_lst_m2d_1/while/Identity:output:0"W
%model_conv_lst_m2d_1_while_identity_1.model/conv_lst_m2d_1/while/Identity_1:output:0"W
%model_conv_lst_m2d_1_while_identity_2.model/conv_lst_m2d_1/while/Identity_2:output:0"W
%model_conv_lst_m2d_1_while_identity_3.model/conv_lst_m2d_1/while/Identity_3:output:0"W
%model_conv_lst_m2d_1_while_identity_4.model/conv_lst_m2d_1/while/Identity_4:output:0"W
%model_conv_lst_m2d_1_while_identity_5.model/conv_lst_m2d_1/while/Identity_5:output:0"?
=model_conv_lst_m2d_1_while_model_conv_lst_m2d_1_strided_slice?model_conv_lst_m2d_1_while_model_conv_lst_m2d_1_strided_slice_0"z
:model_conv_lst_m2d_1_while_split_1_readvariableop_resource<model_conv_lst_m2d_1_while_split_1_readvariableop_resource_0"z
:model_conv_lst_m2d_1_while_split_2_readvariableop_resource<model_conv_lst_m2d_1_while_split_2_readvariableop_resource_0"v
8model_conv_lst_m2d_1_while_split_readvariableop_resource:model_conv_lst_m2d_1_while_split_readvariableop_resource_0"?
{model_conv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor}model_conv_lst_m2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lst_m2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2b
/model/conv_lst_m2d_1/while/split/ReadVariableOp/model/conv_lst_m2d_1/while/split/ReadVariableOp2f
1model/conv_lst_m2d_1/while/split_1/ReadVariableOp1model/conv_lst_m2d_1/while/split_1/ReadVariableOp2f
1model/conv_lst_m2d_1/while/split_2/ReadVariableOp1model/conv_lst_m2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_27534

inputs8
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_27408*
condR
while_cond_27407*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30732

inputs
states_0
states_18
split_readvariableop_resource:@?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29067
inputs_08
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilex

zeros_like	ZerosLikeinputs_0*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_28941*
condR
while_cond_28940*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:h d
>
_output_shapes,
*:(????????????????????
"
_user_specified_name
inputs/0
?
?
0__inference_conv_lstm_cell_1_layer_call_fn_30657

inputs
states_0
states_1"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????@:???????????@:???????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_262142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????@:???????????@:???????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?	
?
.__inference_conv_lst_m2d_1_layer_call_fn_29551

inputs"
unknown:@?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_272922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????@
 
_user_specified_nameinputs
?f
?
while_body_29161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?p
?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29507

inputs8
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs*
T0*>
_output_shapes,
*:(????????????????????2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices}
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????2
Sums
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_8y
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1|
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :???????????@:???????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_29381*
condR
while_cond_29380*_
output_shapesN
L: : : : :???????????@:???????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*>
_output_shapes,
*:(????????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(????????????????????@2
transpose_1?
IdentityIdentitytranspose_1:y:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identity?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?E
?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30548

inputs
states_0
states_18
split_readvariableop_resource:?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution?
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*1
_output_shapes
:???????????@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*1
_output_shapes
:???????????@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*1
_output_shapes
:???????????@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
convolution_7y
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1f
MulMuladd:z:0Const:output:0*
T0*1
_output_shapes
:???????????@2
Muln
Add_1AddV2Mul:z:0Const_1:output:0*
T0*1
_output_shapes
:???????????@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3n
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*1
_output_shapes
:???????????@2
Mul_1p
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*1
_output_shapes
:???????????@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_1p
mul_2Mulclip_by_value_1:z:0states_1*
T0*1
_output_shapes
:???????????@2
mul_2
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
add_4[
ReluRelu	add_4:z:0*
T0*1
_output_shapes
:???????????@2
Relux
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*1
_output_shapes
:???????????@2
mul_3i
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:???????????@2
add_5
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5n
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*1
_output_shapes
:???????????@2
Mul_4p
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*1
_output_shapes
:???????????@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
clip_by_value_2_
Relu_1Relu	add_5:z:0*
T0*1
_output_shapes
:???????????@2
Relu_1|
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
mul_5n
IdentityIdentity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityr

Identity_1Identity	mul_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_1r

Identity_2Identity	add_5:z:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity_2?
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:???????????@:???????????@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
states/1
?f
?
while_body_29647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_conv_lst_m2d_layer_call_fn_28616

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_268042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:(????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_27707

inputs"
unknown:?$
	unknown_0:@?
	unknown_1:	?$
	unknown_2:@?$
	unknown_3:@?
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_275802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(????????????????????
 
_user_specified_nameinputs
Ѕ
?
conv_lst_m2d_while_body_282396
2conv_lst_m2d_while_conv_lst_m2d_while_loop_counter<
8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations"
conv_lst_m2d_while_placeholder$
 conv_lst_m2d_while_placeholder_1$
 conv_lst_m2d_while_placeholder_2$
 conv_lst_m2d_while_placeholder_33
/conv_lst_m2d_while_conv_lst_m2d_strided_slice_0q
mconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0M
2conv_lst_m2d_while_split_readvariableop_resource_0:?O
4conv_lst_m2d_while_split_1_readvariableop_resource_0:@?C
4conv_lst_m2d_while_split_2_readvariableop_resource_0:	?
conv_lst_m2d_while_identity!
conv_lst_m2d_while_identity_1!
conv_lst_m2d_while_identity_2!
conv_lst_m2d_while_identity_3!
conv_lst_m2d_while_identity_4!
conv_lst_m2d_while_identity_51
-conv_lst_m2d_while_conv_lst_m2d_strided_sliceo
kconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensorK
0conv_lst_m2d_while_split_readvariableop_resource:?M
2conv_lst_m2d_while_split_1_readvariableop_resource:@?A
2conv_lst_m2d_while_split_2_readvariableop_resource:	???'conv_lst_m2d/while/split/ReadVariableOp?)conv_lst_m2d/while/split_1/ReadVariableOp?)conv_lst_m2d/while/split_2/ReadVariableOp?
Dconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?      2F
Dconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0conv_lst_m2d_while_placeholderMconv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????*
element_dtype028
6conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem?
"conv_lst_m2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"conv_lst_m2d/while/split/split_dim?
'conv_lst_m2d/while/split/ReadVariableOpReadVariableOp2conv_lst_m2d_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02)
'conv_lst_m2d/while/split/ReadVariableOp?
conv_lst_m2d/while/splitSplit+conv_lst_m2d/while/split/split_dim:output:0/conv_lst_m2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@:@:@:@*
	num_split2
conv_lst_m2d/while/split?
$conv_lst_m2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d/while/split_1/split_dim?
)conv_lst_m2d/while/split_1/ReadVariableOpReadVariableOp4conv_lst_m2d_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02+
)conv_lst_m2d/while/split_1/ReadVariableOp?
conv_lst_m2d/while/split_1Split-conv_lst_m2d/while/split_1/split_dim:output:01conv_lst_m2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d/while/split_1?
$conv_lst_m2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$conv_lst_m2d/while/split_2/split_dim?
)conv_lst_m2d/while/split_2/ReadVariableOpReadVariableOp4conv_lst_m2d_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)conv_lst_m2d/while/split_2/ReadVariableOp?
conv_lst_m2d/while/split_2Split-conv_lst_m2d/while/split_2/split_dim:output:01conv_lst_m2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d/while/split_2?
conv_lst_m2d/while/convolutionConv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2 
conv_lst_m2d/while/convolution?
conv_lst_m2d/while/BiasAddBiasAdd'conv_lst_m2d/while/convolution:output:0#conv_lst_m2d/while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd?
 conv_lst_m2d/while/convolution_1Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_1?
conv_lst_m2d/while/BiasAdd_1BiasAdd)conv_lst_m2d/while/convolution_1:output:0#conv_lst_m2d/while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_1?
 conv_lst_m2d/while/convolution_2Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_2?
conv_lst_m2d/while/BiasAdd_2BiasAdd)conv_lst_m2d/while/convolution_2:output:0#conv_lst_m2d/while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_2?
 conv_lst_m2d/while/convolution_3Conv2D=conv_lst_m2d/while/TensorArrayV2Read/TensorListGetItem:item:0!conv_lst_m2d/while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_3?
conv_lst_m2d/while/BiasAdd_3BiasAdd)conv_lst_m2d/while/convolution_3:output:0#conv_lst_m2d/while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/BiasAdd_3?
 conv_lst_m2d/while/convolution_4Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_4?
 conv_lst_m2d/while/convolution_5Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_5?
 conv_lst_m2d/while/convolution_6Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_6?
 conv_lst_m2d/while/convolution_7Conv2D conv_lst_m2d_while_placeholder_2#conv_lst_m2d/while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2"
 conv_lst_m2d/while/convolution_7?
conv_lst_m2d/while/addAddV2#conv_lst_m2d/while/BiasAdd:output:0)conv_lst_m2d/while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/addy
conv_lst_m2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const}
conv_lst_m2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_1?
conv_lst_m2d/while/MulMulconv_lst_m2d/while/add:z:0!conv_lst_m2d/while/Const:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul?
conv_lst_m2d/while/Add_1AddV2conv_lst_m2d/while/Mul:z:0#conv_lst_m2d/while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_1?
*conv_lst_m2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*conv_lst_m2d/while/clip_by_value/Minimum/y?
(conv_lst_m2d/while/clip_by_value/MinimumMinimumconv_lst_m2d/while/Add_1:z:03conv_lst_m2d/while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2*
(conv_lst_m2d/while/clip_by_value/Minimum?
"conv_lst_m2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv_lst_m2d/while/clip_by_value/y?
 conv_lst_m2d/while/clip_by_valueMaximum,conv_lst_m2d/while/clip_by_value/Minimum:z:0+conv_lst_m2d/while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2"
 conv_lst_m2d/while/clip_by_value?
conv_lst_m2d/while/add_2AddV2%conv_lst_m2d/while/BiasAdd_1:output:0)conv_lst_m2d/while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_2}
conv_lst_m2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const_2}
conv_lst_m2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_3?
conv_lst_m2d/while/Mul_1Mulconv_lst_m2d/while/add_2:z:0#conv_lst_m2d/while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul_1?
conv_lst_m2d/while/Add_3AddV2conv_lst_m2d/while/Mul_1:z:0#conv_lst_m2d/while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_3?
,conv_lst_m2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d/while/clip_by_value_1/Minimum/y?
*conv_lst_m2d/while/clip_by_value_1/MinimumMinimumconv_lst_m2d/while/Add_3:z:05conv_lst_m2d/while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d/while/clip_by_value_1/Minimum?
$conv_lst_m2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d/while/clip_by_value_1/y?
"conv_lst_m2d/while/clip_by_value_1Maximum.conv_lst_m2d/while/clip_by_value_1/Minimum:z:0-conv_lst_m2d/while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/while/clip_by_value_1?
conv_lst_m2d/while/mul_2Mul&conv_lst_m2d/while/clip_by_value_1:z:0 conv_lst_m2d_while_placeholder_3*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_2?
conv_lst_m2d/while/add_4AddV2%conv_lst_m2d/while/BiasAdd_2:output:0)conv_lst_m2d/while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_4?
conv_lst_m2d/while/ReluReluconv_lst_m2d/while/add_4:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Relu?
conv_lst_m2d/while/mul_3Mul$conv_lst_m2d/while/clip_by_value:z:0%conv_lst_m2d/while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_3?
conv_lst_m2d/while/add_5AddV2conv_lst_m2d/while/mul_2:z:0conv_lst_m2d/while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_5?
conv_lst_m2d/while/add_6AddV2%conv_lst_m2d/while/BiasAdd_3:output:0)conv_lst_m2d/while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/add_6}
conv_lst_m2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d/while/Const_4}
conv_lst_m2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d/while/Const_5?
conv_lst_m2d/while/Mul_4Mulconv_lst_m2d/while/add_6:z:0#conv_lst_m2d/while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Mul_4?
conv_lst_m2d/while/Add_7AddV2conv_lst_m2d/while/Mul_4:z:0#conv_lst_m2d/while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Add_7?
,conv_lst_m2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d/while/clip_by_value_2/Minimum/y?
*conv_lst_m2d/while/clip_by_value_2/MinimumMinimumconv_lst_m2d/while/Add_7:z:05conv_lst_m2d/while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2,
*conv_lst_m2d/while/clip_by_value_2/Minimum?
$conv_lst_m2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d/while/clip_by_value_2/y?
"conv_lst_m2d/while/clip_by_value_2Maximum.conv_lst_m2d/while/clip_by_value_2/Minimum:z:0-conv_lst_m2d/while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2$
"conv_lst_m2d/while/clip_by_value_2?
conv_lst_m2d/while/Relu_1Reluconv_lst_m2d/while/add_5:z:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Relu_1?
conv_lst_m2d/while/mul_5Mul&conv_lst_m2d/while/clip_by_value_2:z:0'conv_lst_m2d/while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/mul_5?
7conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem conv_lst_m2d_while_placeholder_1conv_lst_m2d_while_placeholderconv_lst_m2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype029
7conv_lst_m2d/while/TensorArrayV2Write/TensorListSetItemz
conv_lst_m2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/while/add_8/y?
conv_lst_m2d/while/add_8AddV2conv_lst_m2d_while_placeholder#conv_lst_m2d/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d/while/add_8z
conv_lst_m2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d/while/add_9/y?
conv_lst_m2d/while/add_9AddV22conv_lst_m2d_while_conv_lst_m2d_while_loop_counter#conv_lst_m2d/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d/while/add_9?
conv_lst_m2d/while/IdentityIdentityconv_lst_m2d/while/add_9:z:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity?
conv_lst_m2d/while/Identity_1Identity8conv_lst_m2d_while_conv_lst_m2d_while_maximum_iterations^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_1?
conv_lst_m2d/while/Identity_2Identityconv_lst_m2d/while/add_8:z:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_2?
conv_lst_m2d/while/Identity_3IdentityGconv_lst_m2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lst_m2d/while/NoOp*
T0*
_output_shapes
: 2
conv_lst_m2d/while/Identity_3?
conv_lst_m2d/while/Identity_4Identityconv_lst_m2d/while/mul_5:z:0^conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Identity_4?
conv_lst_m2d/while/Identity_5Identityconv_lst_m2d/while/add_5:z:0^conv_lst_m2d/while/NoOp*
T0*1
_output_shapes
:???????????@2
conv_lst_m2d/while/Identity_5?
conv_lst_m2d/while/NoOpNoOp(^conv_lst_m2d/while/split/ReadVariableOp*^conv_lst_m2d/while/split_1/ReadVariableOp*^conv_lst_m2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
conv_lst_m2d/while/NoOp"`
-conv_lst_m2d_while_conv_lst_m2d_strided_slice/conv_lst_m2d_while_conv_lst_m2d_strided_slice_0"C
conv_lst_m2d_while_identity$conv_lst_m2d/while/Identity:output:0"G
conv_lst_m2d_while_identity_1&conv_lst_m2d/while/Identity_1:output:0"G
conv_lst_m2d_while_identity_2&conv_lst_m2d/while/Identity_2:output:0"G
conv_lst_m2d_while_identity_3&conv_lst_m2d/while/Identity_3:output:0"G
conv_lst_m2d_while_identity_4&conv_lst_m2d/while/Identity_4:output:0"G
conv_lst_m2d_while_identity_5&conv_lst_m2d/while/Identity_5:output:0"j
2conv_lst_m2d_while_split_1_readvariableop_resource4conv_lst_m2d_while_split_1_readvariableop_resource_0"j
2conv_lst_m2d_while_split_2_readvariableop_resource4conv_lst_m2d_while_split_2_readvariableop_resource_0"f
0conv_lst_m2d_while_split_readvariableop_resource2conv_lst_m2d_while_split_readvariableop_resource_0"?
kconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensormconv_lst_m2d_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 2R
'conv_lst_m2d/while/split/ReadVariableOp'conv_lst_m2d/while/split/ReadVariableOp2V
)conv_lst_m2d/while/split_1/ReadVariableOp)conv_lst_m2d/while/split_1/ReadVariableOp2V
)conv_lst_m2d/while/split_2/ReadVariableOp)conv_lst_m2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_26039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_26039___redundant_placeholder03
/while_while_cond_26039___redundant_placeholder13
/while_while_cond_26039___redundant_placeholder23
/while_while_cond_26039___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :???????????@:???????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
:
?f
?
while_body_30313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:@?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:@?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*1
_output_shapes
:???????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*1
_output_shapes
:???????????@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*1
_output_shapes
:???????????@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*1
_output_shapes
:???????????@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1~
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*1
_output_shapes
:???????????@2
	while/Mul?
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*1
_output_shapes
:???????????@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_1?
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*1
_output_shapes
:???????????@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*1
_output_shapes
:???????????@2
while/add_4m

while/ReluReluwhile/add_4:z:0*
T0*1
_output_shapes
:???????????@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*1
_output_shapes
:???????????@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*1
_output_shapes
:???????????@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*1
_output_shapes
:???????????@2
while/Mul_4?
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*1
_output_shapes
:???????????@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*1
_output_shapes
:???????????@2
while/clip_by_value_2q
while/Relu_1Reluwhile/add_5:z:0*
T0*1
_output_shapes
:???????????@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*1
_output_shapes
:???????????@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9k
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1o
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*1
_output_shapes
:???????????@2
while/Identity_5?

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :???????????@:???????????@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????@:73
1
_output_shapes
:???????????@:

_output_shapes
: :

_output_shapes
: 
?

?
%__inference_model_layer_call_fn_27057
input_1"
unknown:?$
	unknown_0:@?
	unknown_1:	?$
	unknown_2:@?$
	unknown_3:@?
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:(????????????????????@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_270422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*>
_output_shapes,
*:(????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:(????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
>
_output_shapes,
*:(????????????????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
R
input_1G
serving_default_input_1:0(????????????????????Y
conv_lst_m2d_1G
StatefulPartitionedCall:0(????????????????????@tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
U__call__
*V&call_and_return_all_conditional_losses
W_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
iter

beta_1

beta_2
	decay
learning_ratemImJmKmLmM mNvOvPvQvRvS vT"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
?
!layer_regularization_losses
regularization_losses

"layers
#metrics
$layer_metrics
	variables
trainable_variables
%non_trainable_variables
U__call__
W_default_save_signature
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
,
\serving_default"
signature_map
?

kernel
recurrent_kernel
bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
*layer_regularization_losses
regularization_losses

+layers
,metrics
-layer_metrics

.states
	variables
trainable_variables
/non_trainable_variables
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?

kernel
recurrent_kernel
 bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
?
4layer_regularization_losses
regularization_losses

5layers
6metrics
7layer_metrics

8states
	variables
trainable_variables
9non_trainable_variables
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,?2conv_lst_m2d/kernel
8:6@?2conv_lst_m2d/recurrent_kernel
 :?2conv_lst_m2d/bias
0:.@?2conv_lst_m2d_1/kernel
::8@?2conv_lst_m2d_1/recurrent_kernel
": ?2conv_lst_m2d_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
;layer_regularization_losses
&regularization_losses

<layers
=metrics
>layer_metrics
'	variables
(trainable_variables
?non_trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
?
@layer_regularization_losses
0regularization_losses

Alayers
Bmetrics
Clayer_metrics
1	variables
2trainable_variables
Dnon_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Etotal
	Fcount
G	variables
H	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
E0
F1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
3:1?2Adam/conv_lst_m2d/kernel/m
=:;@?2$Adam/conv_lst_m2d/recurrent_kernel/m
%:#?2Adam/conv_lst_m2d/bias/m
5:3@?2Adam/conv_lst_m2d_1/kernel/m
?:=@?2&Adam/conv_lst_m2d_1/recurrent_kernel/m
':%?2Adam/conv_lst_m2d_1/bias/m
3:1?2Adam/conv_lst_m2d/kernel/v
=:;@?2$Adam/conv_lst_m2d/recurrent_kernel/v
%:#?2Adam/conv_lst_m2d/bias/v
5:3@?2Adam/conv_lst_m2d_1/kernel/v
?:=@?2&Adam/conv_lst_m2d_1/recurrent_kernel/v
':%?2Adam/conv_lst_m2d_1/bias/v
?2?
%__inference_model_layer_call_fn_27057
%__inference_model_layer_call_fn_27690
%__inference_model_layer_call_fn_27707
%__inference_model_layer_call_fn_27612?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_28145
@__inference_model_layer_call_and_return_conditional_losses_28583
@__inference_model_layer_call_and_return_conditional_losses_27630
@__inference_model_layer_call_and_return_conditional_losses_27648?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_25274input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv_lst_m2d_layer_call_fn_28594
,__inference_conv_lst_m2d_layer_call_fn_28605
,__inference_conv_lst_m2d_layer_call_fn_28616
,__inference_conv_lst_m2d_layer_call_fn_28627?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_28847
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29067
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29287
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29507?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_conv_lst_m2d_1_layer_call_fn_29518
.__inference_conv_lst_m2d_1_layer_call_fn_29529
.__inference_conv_lst_m2d_1_layer_call_fn_29540
.__inference_conv_lst_m2d_1_layer_call_fn_29551?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29773
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29995
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30217
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30439?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_27673input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv_lstm_cell_layer_call_fn_30456
.__inference_conv_lstm_cell_layer_call_fn_30473?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30548
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30623?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_conv_lstm_cell_1_layer_call_fn_30640
0__inference_conv_lstm_cell_1_layer_call_fn_30657?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30732
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30807?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_25274? G?D
=?:
8?5
input_1(????????????????????
? "V?S
Q
conv_lst_m2d_1??<
conv_lst_m2d_1(????????????????????@?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29773? Y?V
O?L
>?;
9?6
inputs/0(????????????????????@

 
p 

 
? "<?9
2?/
0(????????????????????@
? ?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_29995? Y?V
O?L
>?;
9?6
inputs/0(????????????????????@

 
p

 
? "<?9
2?/
0(????????????????????@
? ?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30217? R?O
H?E
7?4
inputs(????????????????????@

 
p 

 
? "<?9
2?/
0(????????????????????@
? ?
I__inference_conv_lst_m2d_1_layer_call_and_return_conditional_losses_30439? R?O
H?E
7?4
inputs(????????????????????@

 
p

 
? "<?9
2?/
0(????????????????????@
? ?
.__inference_conv_lst_m2d_1_layer_call_fn_29518? Y?V
O?L
>?;
9?6
inputs/0(????????????????????@

 
p 

 
? "/?,(????????????????????@?
.__inference_conv_lst_m2d_1_layer_call_fn_29529? Y?V
O?L
>?;
9?6
inputs/0(????????????????????@

 
p

 
? "/?,(????????????????????@?
.__inference_conv_lst_m2d_1_layer_call_fn_29540? R?O
H?E
7?4
inputs(????????????????????@

 
p 

 
? "/?,(????????????????????@?
.__inference_conv_lst_m2d_1_layer_call_fn_29551? R?O
H?E
7?4
inputs(????????????????????@

 
p

 
? "/?,(????????????????????@?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_28847?Y?V
O?L
>?;
9?6
inputs/0(????????????????????

 
p 

 
? "<?9
2?/
0(????????????????????@
? ?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29067?Y?V
O?L
>?;
9?6
inputs/0(????????????????????

 
p

 
? "<?9
2?/
0(????????????????????@
? ?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29287?R?O
H?E
7?4
inputs(????????????????????

 
p 

 
? "<?9
2?/
0(????????????????????@
? ?
G__inference_conv_lst_m2d_layer_call_and_return_conditional_losses_29507?R?O
H?E
7?4
inputs(????????????????????

 
p

 
? "<?9
2?/
0(????????????????????@
? ?
,__inference_conv_lst_m2d_layer_call_fn_28594?Y?V
O?L
>?;
9?6
inputs/0(????????????????????

 
p 

 
? "/?,(????????????????????@?
,__inference_conv_lst_m2d_layer_call_fn_28605?Y?V
O?L
>?;
9?6
inputs/0(????????????????????

 
p

 
? "/?,(????????????????????@?
,__inference_conv_lst_m2d_layer_call_fn_28616?R?O
H?E
7?4
inputs(????????????????????

 
p 

 
? "/?,(????????????????????@?
,__inference_conv_lst_m2d_layer_call_fn_28627?R?O
H?E
7?4
inputs(????????????????????

 
p

 
? "/?,(????????????????????@?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30732? ???
???
*?'
inputs???????????@
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p 
? "???
???
'?$
0/0???????????@
Y?V
)?&
0/1/0???????????@
)?&
0/1/1???????????@
? ?
K__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_30807? ???
???
*?'
inputs???????????@
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p
? "???
???
'?$
0/0???????????@
Y?V
)?&
0/1/0???????????@
)?&
0/1/1???????????@
? ?
0__inference_conv_lstm_cell_1_layer_call_fn_30640? ???
???
*?'
inputs???????????@
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p 
? "??~
%?"
0???????????@
U?R
'?$
1/0???????????@
'?$
1/1???????????@?
0__inference_conv_lstm_cell_1_layer_call_fn_30657? ???
???
*?'
inputs???????????@
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p
? "??~
%?"
0???????????@
U?R
'?$
1/0???????????@
'?$
1/1???????????@?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30548????
???
*?'
inputs???????????
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p 
? "???
???
'?$
0/0???????????@
Y?V
)?&
0/1/0???????????@
)?&
0/1/1???????????@
? ?
I__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_30623????
???
*?'
inputs???????????
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p
? "???
???
'?$
0/0???????????@
Y?V
)?&
0/1/0???????????@
)?&
0/1/1???????????@
? ?
.__inference_conv_lstm_cell_layer_call_fn_30456????
???
*?'
inputs???????????
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p 
? "??~
%?"
0???????????@
U?R
'?$
1/0???????????@
'?$
1/1???????????@?
.__inference_conv_lstm_cell_layer_call_fn_30473????
???
*?'
inputs???????????
_?\
,?)
states/0???????????@
,?)
states/1???????????@
p
? "??~
%?"
0???????????@
U?R
'?$
1/0???????????@
'?$
1/1???????????@?
@__inference_model_layer_call_and_return_conditional_losses_27630? O?L
E?B
8?5
input_1(????????????????????
p 

 
? "<?9
2?/
0(????????????????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_27648? O?L
E?B
8?5
input_1(????????????????????
p

 
? "<?9
2?/
0(????????????????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_28145? N?K
D?A
7?4
inputs(????????????????????
p 

 
? "<?9
2?/
0(????????????????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_28583? N?K
D?A
7?4
inputs(????????????????????
p

 
? "<?9
2?/
0(????????????????????@
? ?
%__inference_model_layer_call_fn_27057? O?L
E?B
8?5
input_1(????????????????????
p 

 
? "/?,(????????????????????@?
%__inference_model_layer_call_fn_27612? O?L
E?B
8?5
input_1(????????????????????
p

 
? "/?,(????????????????????@?
%__inference_model_layer_call_fn_27690? N?K
D?A
7?4
inputs(????????????????????
p 

 
? "/?,(????????????????????@?
%__inference_model_layer_call_fn_27707? N?K
D?A
7?4
inputs(????????????????????
p

 
? "/?,(????????????????????@?
#__inference_signature_wrapper_27673? R?O
? 
H?E
C
input_18?5
input_1(????????????????????"V?S
Q
conv_lst_m2d_1??<
conv_lst_m2d_1(????????????????????@
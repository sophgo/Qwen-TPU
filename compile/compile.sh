#!/bin/bash
set -ex
models=
mode="int8"
num_device=1
quantize_args="--quantize W8BF16"
device_args=""
out_model=qwen-7b.bmodel

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --num_device)
        num_device="$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

out_model='qwen-7b_'$mode'.bmodel'

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model='qwen-7b_'$mode'_'$num_device'dev.bmodel'
fi

outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_0.mlir

model_deploy.py \
    --mlir embedding_0.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --model embedding_0.bmodel

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --mlir embedding_1.mlir

model_deploy.py \
    --mlir embedding_1.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --model embedding_1.bmodel

models=$models' '$outdir'/embedding_0.bmodel '$outdir'/embedding_1.bmodel '

popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    ${quantize_args} \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/qwen_block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for i in {0..31}; do

    model_transform.py \
        --model_name qwen_block_$i \
        --model_def ../../onnx/qwen_block_$i.onnx \
        --mlir qwen_block_$i.mlir

    model_deploy.py \
        --mlir qwen_block_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --model qwen_block_$i.bmodel

    model_transform.py \
        --model_name qwen_block_cache_$i \
        --model_def ../../onnx/qwen_block_cache_$i.onnx \
        --mlir qwen_block_cache_$i.mlir

    model_deploy.py \
        --mlir qwen_block_cache_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --io_alone \
        --model qwen_block_cache_$i.bmodel

    models=${models}${outdir}'/qwen_block_'$i'.bmodel '$outdir'/qwen_block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model

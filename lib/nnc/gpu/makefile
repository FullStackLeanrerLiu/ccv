include ../../config.mk

CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS) -I"3rdparty/cutlass/include"

FLASH_ATTN_SRCS := 3rdparty/flash_attn/flash_api.cu 3rdparty/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu
FLASH_ATTN_SRCS += 3rdparty/flash_attn/layer_norm/ln_api.cu 3rdparty/flash_attn/layer_norm/ln_fwd_256.cu 3rdparty/flash_attn/layer_norm/ln_fwd_512.cu 3rdparty/flash_attn/layer_norm/ln_fwd_768.cu 3rdparty/flash_attn/layer_norm/ln_fwd_1024.cu 3rdparty/flash_attn/layer_norm/ln_fwd_1280.cu 3rdparty/flash_attn/layer_norm/ln_fwd_1536.cu 3rdparty/flash_attn/layer_norm/ln_fwd_2048.cu 3rdparty/flash_attn/layer_norm/ln_fwd_2560.cu 3rdparty/flash_attn/layer_norm/ln_fwd_3072.cu 3rdparty/flash_attn/layer_norm/ln_fwd_4096.cu 3rdparty/flash_attn/layer_norm/ln_fwd_5120.cu 3rdparty/flash_attn/layer_norm/ln_fwd_6144.cu 3rdparty/flash_attn/layer_norm/ln_fwd_7168.cu 3rdparty/flash_attn/layer_norm/ln_fwd_8192.cu

# sm75 (Turing) builds compile ONLY the FP16 fwd + fwd_split kernels: bf16 MMA and
# cp.async are Ampere-only, so the bf16 / bwd sm80 sources can't be built for sm75.
# The SageAttention-style INT8-QK and INT4-QK forward kernels are additionally
# emitted for head dims {64,128,256} (flash_fwd_int8_hdim*_sm75.cu /
# flash_fwd_int4_hdim*_sm75.cu).
FLASH_ATTN_SM75_SRCS := 3rdparty/flash_attn/flash_api.cu $(wildcard 3rdparty/flash_attn/src/flash_fwd_hdim*_fp16_sm75.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim*_fp16_sm75.cu 3rdparty/flash_attn/src/flash_fwd_int8_hdim*_sm75.cu 3rdparty/flash_attn/src/flash_fwd_int4_hdim*_sm75.cu)

SRCS := ccv_nnc_compat.cu ccv_nnc_palettize.cu

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(SRCS)))

FLASH_ATTN_SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(FLASH_ATTN_SRCS)))

FLASH_ATTN_SM75_SRC_OBJS := $(patsubst %.cu,%.o,$(FLASH_ATTN_SM75_SRCS))

.PHONY: release all lib clean libnnc-compat-cuda.o libnnc-compat-cuda-sm80.o libnnc-compat-cuda-sm75.o

release: all

include ../../scheme.mk

all: lib

lib: libnnc-compat-cuda.o

clean:
	rm -f *.o

libnnc-compat-cuda.o: $(SRC_OBJS) $(CUDA_COMPAT_SM80_LIB) $(CUDA_COMPAT_SM75_LIB)
	ld -r $^ -o $@

libnnc-compat-cuda-sm80.o: $(FLASH_ATTN_SRC_OBJS)
	ld -r $^ -o $@

libnnc-compat-cuda-sm75.o: $(FLASH_ATTN_SM75_SRC_OBJS)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.c,,$^) ; do \
		$(NVCC) $(NVFLAGS) -M $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk


# 方便直接编译 CUDA 文件
# 参数： 本脚本文件名 要编译的cuda文件 编译后保存的文件
# 适用于 CUDA 8.0， GTX 1080(Ti)

nvcc -arch sm_61 ${1} -o ${2}

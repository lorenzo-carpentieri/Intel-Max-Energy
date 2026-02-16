
module load oneapi
module load geopm-runtime
module load pti-gpu
module load cmake
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE # Intel Max as single GPU with two subdevice
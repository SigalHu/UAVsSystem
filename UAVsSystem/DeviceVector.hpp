#include "common.h"
#include "cuda_task.h"
#include "DeviceVector.h"

template<class _T, class _Alloc>
const thrust::device_ptr<void> DeviceVector<_T, _Alloc>::getPtr(){
	return this->data();
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector()
:DeviceManager(), device_vector(){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(const size_t &n)
:DeviceManager(), device_vector(n){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const size_t &n)
: DeviceManager(deviceId), device_vector(n){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const size_t &n, const _T &value)
: DeviceManager(deviceId), device_vector(n, value){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(const DeviceVector &v)
: DeviceManager(v.getDeviceId()), device_vector(v){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(const DeviceVector &v, const size_t &start, const size_t &len)
: DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::DeviceVector(DeviceVector &&v)
: DeviceManager(v.getDeviceId()), device_vector(move(v)){
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>& DeviceVector<_T, _Alloc>::operator=(const DeviceVector &v){
	if (getDeviceId() != v.getDeviceId())
		throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_CurFunctionName(), getDeviceIdString(),
		StringUtils::format(getDeviceIdString().append("(=%d) is not equal to another ").append(getDeviceIdString()).append("(=%d)."),
		getDeviceId(), v.getDeviceId()));
	this->switch2Device();
	device_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>& DeviceVector<_T, _Alloc>::operator=(DeviceVector &&v){
	if (getDeviceId() != v.getDeviceId())
		throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_CurFunctionName(), getDeviceIdString(),
		StringUtils::format(getDeviceIdString().append("(=%d) is not equal to another ").append(getDeviceIdString()).append("(=%d)."),
		getDeviceId(), v.getDeviceId()));
	this->switch2Device();
	device_vector::operator=(move(v));
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v)
:DeviceManager(v.getDeviceId()), device_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
: DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>& DeviceVector<_T, _Alloc>::operator=(const DeviceVector<_OtherT, _OtherAlloc> &v){
	if (getDeviceId() != v.getDeviceId())
		throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_CurFunctionName(), getDeviceIdString(),
		StringUtils::format(getDeviceIdString().append("(=%d) is not equal to another ").append(getDeviceIdString()).append("(=%d)."),
		getDeviceId(), v.getDeviceId()));
	this->switch2Device();
	device_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const std::vector<_OtherT, _OtherAlloc> &v)
: DeviceManager(), device_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const std::vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT, _OtherAlloc> &v)
: DeviceManager(deviceId), device_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>& DeviceVector<_T, _Alloc>::operator=(const std::vector<_OtherT, _OtherAlloc> &v){
	this->switch2Device();
	device_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v)
: DeviceManager(), device_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v)
: DeviceManager(deviceId), device_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>::DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
DeviceVector<_T, _Alloc>& DeviceVector<_T, _Alloc>::operator=(const HostVector<_OtherT, _OtherAlloc> &v){
	this->switch2Device();
	device_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
DeviceVector<_T, _Alloc>::~DeviceVector(){
}

template<class _T, class _Alloc>
void DeviceVector<_T, _Alloc>::call(const CudaTask &cudaTask, std::initializer_list<std::shared_ptr<DeviceManager>> others){
	std::vector<thrust::device_ptr<void>> otherPtrs;
	otherPtrs.reserve(others.size());

	for (const std::shared_ptr<DeviceManager> &item : others){
		if (getDeviceId() != item->getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_CurFunctionName(), getDeviceIdString(),
			StringUtils::format(getDeviceIdString().append("(=%d) is not equal to another ").append(getDeviceIdString()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		otherPtrs.emplace_back(item->getPtr());
	}
	this->switch2Device();
	cudaTask(this->data(), otherPtrs);
}
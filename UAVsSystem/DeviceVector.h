#pragma once
#include <vector>
#include "thrust\device_vector.h"
//#include "HostVector.h"
#include "cuda_utils.h"
#include "common.h"
using namespace std;
using namespace thrust;

template<class _T, class _Alloc>
class HostVector;

template<class _T, class _Alloc = device_malloc_allocator<_T>>
class DeviceVector :public device_vector < _T, _Alloc > {
private:
	static vector<unsigned int> useCount;
	const unsigned int deviceId;
private:
	void setDeviceId(const unsigned int &deviceId){
		if (deviceId >= DeviceVector::useCount.size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			StringUtils::format(SystemCodeEnum::OUT_OF_RANGE.getInfo(),
			MacroUtils_ClassName(*this).c_str(), MacroUtils_FunctionName().c_str(), MacroUtils_VariableName(deviceId).c_str(),
			MacroUtils_VariableName(deviceId).append(" must be less than the number of GPU devices.").c_str()));
		this->deviceId = deviceId;
	}
public:
	DeviceVector(){
		setDeviceId(0);
		DeviceVector::useCount[getDeviceId()]++;
	}

	DeviceVector(const size_t &n) :device_vector(n){
		setDeviceId(0);
		DeviceVector::useCount[getDeviceId()]++;
	}

	DeviceVector(const size_t &n, const _T &value) :device_vector(n, value){
		setDeviceId(0);
		DeviceVector::useCount[getDeviceId()]++;
	}

	DeviceVector(const DeviceVector &v) :device_vector(v){
		setDeviceId(v.getDeviceId());
		DeviceVector::useCount[getDeviceId()]++;
	}

	DeviceVector(DeviceVector &&v) :device_vector(move(v)){
		setDeviceId(v.getDeviceId());
		DeviceVector::useCount[getDeviceId()]++;
	}

	DeviceVector &operator=(const DeviceVector &v){
		device_vector::operator=(v);
		return *this;
	}

	DeviceVector &operator=(DeviceVector &&v){
		device_vector::operator=(move(v));
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const DeviceVector<_OtherT, _OtherAlloc> &v){
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const vector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const vector<_OtherT, _OtherAlloc> &v){
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const HostVector<_OtherT, _OtherAlloc> &v){
		device_vector::operator=(v);
		return *this;
	}

	template<class InputIterator>
	DeviceVector(InputIterator first, InputIterator last) : device_vector(first, last){}

	~DeviceVector(){
	}

	unsigned int getDeviceId() const{
		return this->deviceId;
	}
};

template<class _T, class _Alloc>
vector<unsigned int> DeviceVector<_T, _Alloc>::useCount(CudaUtils::getDeviceCount(), 0);


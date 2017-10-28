#pragma once
#include <vector>
#include "thrust\device_vector.h"
//#include "HostVector.h"
#include "cuda_utils.h"
#include "common.h"
#include "DeviceManager.h"
using namespace std;
using namespace thrust;

template<class _T, class _Alloc>
class HostVector;

template<class _T, class _Alloc = device_malloc_allocator<_T>>
class DeviceVector :public device_vector < _T, _Alloc >, virtual public DeviceManager {
private:
	void* getPtr() override{
		return this->data();
	}
public:
	DeviceVector()
		:DeviceManager(), device_vector(){}

	DeviceVector(const size_t &n)
		:DeviceManager(), device_vector(n){}

	DeviceVector(const unsigned int &deviceId, const size_t &n)
		:DeviceManager(deviceId), device_vector(n){}

	DeviceVector(const unsigned int &deviceId, const size_t &n, const _T &value)
		:DeviceManager(deviceId), device_vector(n, value){}

	DeviceVector(const DeviceVector &v)
		:DeviceManager(v.getDeviceId()), device_vector(v){}

	DeviceVector(const DeviceVector &v, const size_t &start, const size_t &len)
		:DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){}

	DeviceVector(DeviceVector &&v)
		:DeviceManager(v.getDeviceId()), device_vector(move(v)){}

	DeviceVector &operator=(const DeviceVector &v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_FunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		device_vector::operator=(v);
		return *this;
	}

	DeviceVector &operator=(DeviceVector &&v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_FunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		device_vector::operator=(move(v));
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v)
		:DeviceManager(v.getDeviceId()), device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
		: DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const DeviceVector<_OtherT, _OtherAlloc> &v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_FunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const vector<_OtherT, _OtherAlloc> &v)
		: DeviceManager(), device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
		: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const vector<_OtherT, _OtherAlloc> &v) 
		: DeviceManager(deviceId), device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
		: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const vector<_OtherT, _OtherAlloc> &v){
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v) 
		: DeviceManager(), device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
		: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v) 
		: DeviceManager(deviceId), device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len)
		: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT, class _OtherAlloc>
	DeviceVector &operator=(const HostVector<_OtherT, _OtherAlloc> &v){
		device_vector::operator=(v);
		return *this;
	}

	//template<class InputIterator>
	//DeviceVector(InputIterator first, InputIterator last) : DeviceManager(), device_vector(first, last){
	//}

	~DeviceVector(){
	}

	void call(const CudaTask& cudaTask, initializer_list<shared_ptr<DeviceManager>> others = {}){
		vector<void*> otherPtrs;
		otherPtrs.reserve(others.size());
		for (const shared_ptr<DeviceManager> &item : others){
			if (getDeviceId() != item->getDeviceId())
				throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(*this), MacroUtils_FunctionName(), getDeviceIdStr(),
				StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
				getDeviceId(), v.getDeviceId()));
			otherPtrs.emplace_back(item->getPtr());
		}

		cudaTask(this->getPtr(), otherPtrs);
	}
};


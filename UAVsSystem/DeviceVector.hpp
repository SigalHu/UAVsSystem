#pragma once
#include <vector>
#include <memory>
#include <initializer_list>
#include "thrust\device_vector.h"
#include "DeviceManager.h"
#include "common.h"
#include "cuda_task.h"
#include "HostVector.h"

template<class _T>
class DeviceVector :public thrust::device_vector< _T>, virtual public DeviceManager {
private:
	static std::string getClassName(){
		return MacroUtils_ClassName(DeviceVector);
	}

	const thrust::device_ptr<void> getPtr() override{
		return this->data();
	}
public:
	DeviceVector() 
		: DeviceManager(), device_vector(){}
	DeviceVector(const size_t &n) 
		: DeviceManager(), device_vector(n){}
	DeviceVector(const unsigned int &deviceId, const size_t &n)
		: DeviceManager(deviceId), device_vector(n){}
	DeviceVector(const unsigned int &deviceId, const size_t &n, const _T &value)
		: DeviceManager(deviceId), device_vector(n, value){}
	DeviceVector(const DeviceVector &v)
		: DeviceManager(v.getDeviceId()), device_vector(v){}
	DeviceVector(const DeviceVector &v, const size_t &start, const size_t &len)
		: DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){}
	DeviceVector(DeviceVector &&v)
		: DeviceManager(v.getDeviceId()), device_vector(move(v)){}

	DeviceVector& operator=(const DeviceVector &v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		this->switch2Device();
		device_vector::operator=(v);
		return *this;
	}
	DeviceVector& operator=(DeviceVector &&v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		this->switch2Device();
		device_vector::operator=(move(v));
		return *this;
	}

	template<class _OtherT>
	DeviceVector(const DeviceVector<_OtherT> &v)
		: DeviceManager(v.getDeviceId()), device_vector(v){}
	template<class _OtherT>
	DeviceVector(const DeviceVector<_OtherT> &v, const size_t &start, const size_t &len)
		: DeviceManager(v.getDeviceId()), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT>
	DeviceVector& operator=(const DeviceVector<_OtherT> &v){
		if (getDeviceId() != v.getDeviceId())
			throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), getDeviceIdStr(),
			StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
			getDeviceId(), v.getDeviceId()));
		this->switch2Device();
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT>
	DeviceVector(const std::vector<_OtherT> &v)
		: DeviceManager(), device_vector(v){}
	template<class _OtherT>
	DeviceVector(const std::vector<_OtherT> &v, const size_t &start, const size_t &len)
		: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){}
	template<class _OtherT>
	DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT> &v)
		: DeviceManager(deviceId), device_vector(v){}
	template<class _OtherT>
	DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT> &v, const size_t &start, const size_t &len)
		: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT>
	DeviceVector& operator=(const std::vector<_OtherT> &v){
		this->switch2Device();
		device_vector::operator=(v);
		return *this;
	}

	template<class _OtherT>
	DeviceVector(const HostVector<_OtherT> &v)
		: DeviceManager(), device_vector(v){}
	template<class _OtherT>
	DeviceVector(const HostVector<_OtherT> &v, const size_t &start, const size_t &len)
		: DeviceManager(), device_vector(v.begin() + start, v.begin() + start + len){}
	template<class _OtherT>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT> &v)
		: DeviceManager(deviceId), device_vector(v){}
	template<class _OtherT>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT> &v, const size_t &start, const size_t &len)
		: DeviceManager(deviceId), device_vector(v.begin() + start, v.begin() + start + len){}

	template<class _OtherT>
	DeviceVector& operator=(const HostVector<_OtherT> &v){
		this->switch2Device();
		device_vector::operator=(v);
		return *this;
	}

	//template<class InputIterator>
	//DeviceVector(InputIterator first, InputIterator last) : DeviceManager(), device_vector(first, last){
	//}

	~DeviceVector(){}

	void call(const CudaTask &cudaTask, std::initializer_list<std::shared_ptr<DeviceManager>> others = {}){
		std::vector<thrust::device_ptr<void>> otherPtrs;
		otherPtrs.reserve(others.size());

		for (const std::shared_ptr<DeviceManager> &item : others){
			if (getDeviceId() != item->getDeviceId())
				throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), getDeviceIdStr(),
				StringUtils::format(getDeviceIdStr().append("(=%d) is not equal to another ").append(getDeviceIdStr()).append("(=%d)."),
				getDeviceId(), v.getDeviceId()));
			otherPtrs.emplace_back(item->getPtr());
		}
		this->switch2Device();
		cudaTask(this->data(), otherPtrs);
	}
};
#pragma once
#include <vector>
#include <memory>
#include <initializer_list>
#include "thrust\device_vector.h"
#include "DeviceManager.h"

template<class _OtherT, class _OtherAlloc>
class HostVector;
class CudaTask;

template<class _T, class _Alloc = thrust::device_malloc_allocator<_T>>
class DeviceVector :public thrust::device_vector < _T, _Alloc >, virtual public DeviceManager {
private:
	const thrust::device_ptr<void> getPtr() override;
public:
	DeviceVector();
	DeviceVector(const size_t &n);
	DeviceVector(const unsigned int &deviceId, const size_t &n);
	DeviceVector(const unsigned int &deviceId, const size_t &n, const _T &value);
	DeviceVector(const DeviceVector &v);
	DeviceVector(const DeviceVector &v, const size_t &start, const size_t &len);
	DeviceVector(DeviceVector &&v);

	DeviceVector& operator=(const DeviceVector &v);
	DeviceVector& operator=(DeviceVector &&v);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector& operator=(const DeviceVector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const std::vector<_OtherT, _OtherAlloc> &v);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const std::vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT, _OtherAlloc> &v);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const std::vector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector& operator=(const std::vector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v);
	template<class _OtherT, class _OtherAlloc>
	DeviceVector(const unsigned int &deviceId, const HostVector<_OtherT, _OtherAlloc> &v, const size_t &start, const size_t &len);

	template<class _OtherT, class _OtherAlloc>
	DeviceVector& operator=(const HostVector<_OtherT, _OtherAlloc> &v);

	//template<class InputIterator>
	//DeviceVector(InputIterator first, InputIterator last) : DeviceManager(), device_vector(first, last){
	//}

	~DeviceVector();

	void call(const CudaTask &cudaTask, std::initializer_list<std::shared_ptr<DeviceManager>> others = {});
};


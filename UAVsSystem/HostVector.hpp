#pragma once
#include <vector>
#include "thrust\host_vector.h"
#include "DeviceVector.h"

template<class _T>
class HostVector :public thrust::host_vector<_T> {
public:
	HostVector(){}
	HostVector(const size_t &n)
		: host_vector(n){}
	HostVector(const size_t &n, const _T &value)
		: host_vector(n, value){}
	HostVector(const HostVector &v)
		: host_vector(v){}
	HostVector(HostVector &&v)
		: host_vector(move(v)){}

	HostVector<_T>& operator=(const HostVector &v){
		host_vector::operator=(v);
		return *this;
	}
	HostVector<_T>& operator=(HostVector &&v){
		host_vector::operator=(move(v));
		return *this;
	}

	template<class _OtherT>
	HostVector(const HostVector<_OtherT> &v)
		: host_vector(v){}

	template<class _OtherT>
	HostVector<_T>& operator=(const HostVector<_OtherT> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class _OtherT>
	HostVector(const std::vector<_OtherT> &v)
		: host_vector(v){}

	template<class _OtherT>
	HostVector<_T>& operator=(const std::vector<_OtherT> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class _OtherT>
	HostVector(const DeviceVector<_OtherT> &v)
		: host_vector(v){}

	template<class _OtherT>
	HostVector<_T>& operator=(const DeviceVector<_OtherT> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class InputIterator>
	HostVector(InputIterator first, InputIterator last)
		: host_vector(first, last){}
};
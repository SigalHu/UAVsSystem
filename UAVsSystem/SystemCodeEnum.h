#pragma once

class SystemCode;

class SystemCodeEnum final{
public:
	static const SystemCode CUDA_RUNTIME_ERROR;
	static const SystemCode CUDA_CALL_ERROR;

	static const SystemCode NULL_POINTER;
	static const SystemCode OUT_OF_RANGE;
	static const SystemCode NOT_EQUAL;
private:
	SystemCodeEnum() = default;
	SystemCodeEnum(const SystemCodeEnum &) = default;
	virtual ~SystemCodeEnum() = 0;
};


#pragma once

#include <exception>
#include "common_definition.h"
using namespace std;

class SystemException final:public exception {
private:
	const SystemCode &_Code;
private:
	SystemException() = default;
	SystemException(const SystemException &) = default;
public:
	SystemException(const SystemCode &code, const string &info);
	~SystemException();

	unsigned int code() const;
};


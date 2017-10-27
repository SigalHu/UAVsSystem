#pragma once

#include <exception>
#include "common_definition.h"
using namespace std;

class SystemException final:public exception {
private:
	const SystemCode &_Code;
private:
	SystemException() = default;

	string formatException(const SystemCode &code, const string &className, const string &functionName, const string &variableName, const string &detail) const;
public:
	SystemException(const SystemCode &code, const string &info);
	SystemException(const SystemCode &code, const string &className, const string &functionName, const string &variableName, const string &detail);
	~SystemException();

	unsigned int code() const;
};


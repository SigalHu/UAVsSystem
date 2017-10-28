#pragma once
#include <string>
#include <exception>
#include "SystemCode.h"

class SystemException final :public std::exception {
private:
	const SystemCode &_Code;
	std::string _ClassName;
	std::string _FunctionName;
	std::string _VariableName;
	std::string _Detail;
private:
	SystemException() = default;
public:
	SystemException(const SystemCode &code, const std::string &className, const std::string &functionName, const std::string &variableName, const std::string &detail);
	~SystemException();

	unsigned int code() const;
	const char* what() const override;

	std::string getClassName() const;
	void setClassName(const std::string &className);

	std::string getFunctionName() const;
	void setFunctionName(const std::string &functionName);

	std::string getVariableName() const;
	void setVariableName(const std::string &variableName);

	std::string getDetail() const;
	void setDetail(const std::string &detail);
};


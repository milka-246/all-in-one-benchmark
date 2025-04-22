#pragma once

/*
	An abstract class from which all 
	benchmarks should be inherited.
*/

#include<string>

class Benchmark {
public:
	virtual std::string name() const = 0;
	virtual void run() = 0;
	virtual ~Benchmark() = default;
};